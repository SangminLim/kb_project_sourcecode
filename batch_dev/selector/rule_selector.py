from __future__ import annotations

from typing import Any, Dict, List, Optional


def _normalize_text(text: str) -> str:
    return (text or "").upper().replace(" ", "")


def _contains_any(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    normalized = _normalize_text(text)
    return any(_normalize_text(keyword) in normalized for keyword in keywords)


def _contains_excluded(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return False
    normalized = _normalize_text(text)
    return any(_normalize_text(keyword) in normalized for keyword in keywords)


def _table_by_name(erwin_meta: Dict[str, Any], table_name: str) -> Optional[Dict[str, Any]]:
    for table in erwin_meta.get("tables", []):
        if table.get("table_name") == table_name:
            return table
    return None


def _tables_by_role(erwin_meta: Dict[str, Any], role: str) -> List[Dict[str, Any]]:
    return [
        table
        for table in erwin_meta.get("tables", [])
        if table.get("table_role") == role
    ]


def _column_by_role(table: Dict[str, Any], role: str) -> Optional[str]:
    for column in table.get("columns", []):
        if column.get("role") == role:
            return column.get("column_name")
    return None


def _has_column(table: Dict[str, Any], column_name: str) -> bool:
    return any(column.get("column_name") == column_name for column in table.get("columns", []))


def _extract_table_names_from_request(request_text: str, erwin_meta: Dict[str, Any]) -> List[str]:
    found: List[str] = []
    normalized_request = _normalize_text(request_text)

    for table in erwin_meta.get("tables", []):
        table_name = table.get("table_name", "")
        table_kor_name = table.get("table_kor_name", "")
        aliases = table.get("aliases", [])

        candidates = [table_name, table_kor_name, *aliases]
        if any(_normalize_text(candidate) in normalized_request for candidate in candidates if candidate):
            found.append(table_name)

    return found


def _has_required_table_roles(erwin_meta: Dict[str, Any], required_roles: List[str]) -> bool:
    available_roles = {
        table.get("table_role")
        for table in erwin_meta.get("tables", [])
        if table.get("table_role")
    }
    return all(role in available_roles for role in required_roles)


def select_rule(
    request_text: str,
    erwin_meta: Dict[str, Any],
    rule_catalog: Dict[str, Any],
) -> Dict[str, Any]:
    """
    업무명 하드코딩이 아니라 처리 패턴을 선택한다.
    테이블/컬럼/JOIN은 ERWIN meta에서 해석한다.
    """
    rules = sorted(
        rule_catalog.get("rules", []),
        key=lambda rule: int(rule.get("priority", 0)),
        reverse=True,
    )

    mentioned_tables = _extract_table_names_from_request(request_text, erwin_meta)
    mentioned_table_objs = [
        table
        for table_name in mentioned_tables
        if (table := _table_by_name(erwin_meta, table_name))
    ]
    mentioned_roles = {
        table.get("table_role")
        for table in mentioned_table_objs
        if table.get("table_role")
    }

    for rule in rules:
        match = rule.get("match") or {}

        if not _contains_any(request_text, match.get("required_any", [])):
            continue

        if _contains_excluded(request_text, match.get("exclude_any", [])):
            continue

        required_table_roles = match.get("required_table_roles", [])
        if required_table_roles:
            if mentioned_roles:
                if not all(role in mentioned_roles for role in required_table_roles):
                    continue
            elif not _has_required_table_roles(erwin_meta, required_table_roles):
                continue

        return rule

    raise ValueError("요청서와 ERWIN 메타에 맞는 배치 패턴 rule을 찾지 못했습니다.")


def _resolve_ledger_table(erwin_meta: Dict[str, Any], request_text: str) -> Dict[str, Any]:
    mentioned_names = _extract_table_names_from_request(request_text, erwin_meta)

    for table_name in mentioned_names:
        table = _table_by_name(erwin_meta, table_name)
        if table and table.get("table_role") == "transaction_ledger":
            return table

    ledger_tables = _tables_by_role(erwin_meta, "transaction_ledger")
    if not ledger_tables:
        raise ValueError("ERWIN 메타에서 transaction_ledger 테이블을 찾지 못했습니다.")

    return ledger_tables[0]


def _resolve_classification_tables(
    erwin_meta: Dict[str, Any],
    request_text: str,
    ledger_table_name: str,
) -> List[Dict[str, Any]]:
    mentioned_names = set(_extract_table_names_from_request(request_text, erwin_meta))
    classification_tables = _tables_by_role(erwin_meta, "classification_master")

    # 요청서에 참조 테이블이 명시되어 있으면 그 테이블만 우선 사용한다.
    explicitly_mentioned = [
        table
        for table in classification_tables
        if table.get("table_name") in mentioned_names
    ]
    if explicitly_mentioned:
        return explicitly_mentioned

    # 아니면 relation으로 연결된 classification_master를 사용한다.
    related_table_names = {
        relation.get("right_table")
        for relation in erwin_meta.get("relations", [])
        if relation.get("left_table") == ledger_table_name
    }

    related = [
        table
        for table in classification_tables
        if table.get("table_name") in related_table_names
    ]

    if not related:
        raise ValueError("거래 원장과 연결된 classification_master 테이블을 찾지 못했습니다.")

    return related


def _build_select_clause(
    ledger_table: Dict[str, Any],
    request_text: str,
) -> str:
    """
    출력 컬럼은 요청서에 명시된 컬럼을 우선 사용한다.
    요청서에 없으면 role 기반 기본 컬럼을 사용한다.
    MERCHANT_TYPE은 classification 결과 컬럼이므로 마지막에 추가한다.
    """
    columns = [column.get("column_name") for column in ledger_table.get("columns", [])]
    normalized_request = _normalize_text(request_text)

    requested_columns = [
        column
        for column in columns
        if column and _normalize_text(column) in normalized_request
    ]

    if not requested_columns:
        role_order = [
            "transaction_date",
            "customer_id",
            "merchant_id",
            "amount",
            "base_month",
        ]
        requested_columns = []

        # PK 먼저
        for pk in ledger_table.get("primary_keys", []):
            if pk not in requested_columns:
                requested_columns.append(pk)

        # role 기반 중요 컬럼
        for role in role_order:
            column_name = _column_by_role(ledger_table, role)
            if column_name and column_name not in requested_columns:
                requested_columns.append(column_name)

    # 요청서에 MERCHANT_TYPE이 있거나 분류 패턴이면 항상 추가
    select_lines = [f"    L.{column}" for column in requested_columns]
    select_lines.append("    CASE")
    select_lines.append("{{ merchant_type_case_inner }}")
    select_lines.append("    END AS MERCHANT_TYPE")

    return ",\n".join(select_lines)


def _build_where_base_conditions(
    ledger_table: Dict[str, Any],
    request_text: str,
) -> List[str]:
    conditions: List[str] = []

    base_month_column = _column_by_role(ledger_table, "base_month")
    if base_month_column:
        conditions.append(f"    L.{base_month_column} = :base_ym")

    cancel_column = _column_by_role(ledger_table, "cancel_flag")
    if cancel_column:
        conditions.append(f"    L.{cancel_column} = 'N'")

    return conditions


def _make_alias(index: int, table: Dict[str, Any]) -> str:
    value = table.get("classification_value") or table.get("table_name") or f"C{index}"
    letters = "".join(ch for ch in value if ch.isalpha())
    if letters:
        return letters[0].upper() + str(index)
    return f"C{index}"


def build_ledger_extract_context(
    erwin_meta: Dict[str, Any],
    request_text: str = "",
) -> Dict[str, str]:
    """
    SQL 템플릿에 넘길 최종 clause들을 ERWIN meta로 생성한다.

    템플릿에는 SELECT 컬럼명, 테이블명, WHERE 조건을 직접 박지 않는다.
    여기에서 table_role, column role, relations를 읽어서 생성한다.
    """
    ledger_table = _resolve_ledger_table(erwin_meta, request_text)
    ledger_table_name = ledger_table["table_name"]
    classification_tables = _resolve_classification_tables(erwin_meta, request_text, ledger_table_name)

    select_clause = _build_select_clause(ledger_table, request_text)
    from_clause = f"{ledger_table_name} L"

    join_lines: List[str] = []
    case_lines: List[str] = []
    exists_conditions: List[str] = []

    classification_table_names = {table.get("table_name") for table in classification_tables}

    relation_index = 0
    for relation in erwin_meta.get("relations", []):
        if relation.get("left_table") != ledger_table_name:
            continue

        right_table_name = relation.get("right_table")
        if right_table_name not in classification_table_names:
            continue

        classification_table = _table_by_name(erwin_meta, right_table_name)
        if not classification_table:
            continue

        alias = _make_alias(relation_index, classification_table)
        relation_index += 1

        join_conditions: List[str] = []

        for join_col in relation.get("join_columns", []):
            left_col = join_col.get("left_column")
            right_col = join_col.get("right_column")
            join_conditions.append(f"L.{left_col} = {alias}.{right_col}")

        effective_date = relation.get("effective_date") or {}
        transaction_date_col = effective_date.get("transaction_date_column")
        start_col = effective_date.get("start_column")
        end_col = effective_date.get("end_column")

        if transaction_date_col and start_col:
            join_conditions.append(f"L.{transaction_date_col} >= {alias}.{start_col}")

        if transaction_date_col and end_col:
            join_conditions.append(
                f"({alias}.{end_col} IS NULL OR L.{transaction_date_col} <= {alias}.{end_col})"
            )

        use_flag_col = _column_by_role(classification_table, "use_flag")
        if use_flag_col:
            join_conditions.append(f"{alias}.{use_flag_col} = 'Y'")

        join_sql = (
            f"LEFT JOIN {right_table_name} {alias}\n"
            f"    ON " + "\n   AND ".join(join_conditions)
        )
        join_lines.append(join_sql)

        merchant_col = _column_by_role(classification_table, "merchant_id") or "MERCHANT_ID"
        classification_value = classification_table.get("classification_value") or right_table_name

        case_lines.append(f"        WHEN {alias}.{merchant_col} IS NOT NULL THEN '{classification_value}'")
        exists_conditions.append(f"    {alias}.{merchant_col} IS NOT NULL")

    if not join_lines:
        raise ValueError("ERWIN relations 기준으로 JOIN 절을 생성하지 못했습니다.")

    where_conditions = _build_where_base_conditions(ledger_table, request_text)
    if exists_conditions:
        where_conditions.append("    (\n" + "\n    OR ".join(exists_conditions) + "\n    )")

    return {
        "select_clause": select_clause.replace("{{ merchant_type_case_inner }}", "\n".join(case_lines)),
        "from_clause": from_clause,
        "join_clause": "\n".join(join_lines),
        "where_clause": "\n  AND ".join(where_conditions),
    }
