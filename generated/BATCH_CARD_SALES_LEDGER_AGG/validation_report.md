# 🔍 배치 생성 검증 리포트

- 최종 상태: **✅ PASS WITH WARNINGS**
- 점수: **0.82**
- 배치 유형: **aggregation_to_table**
- 검증정책: **practical-scoring-v3-nonblocking-warn**

## 요약
소득공제 월별 통합 집계 배치가 생성되었으며, SQL은 월별 집계와 Delete Insert 방식을 반영하고 있으나 JOIN/GROUP BY/INSERT/DELETE 구조로 인해 운영 검증 부담이 크다. 운영 반영 전 검토사항이 있습니다.

## 배치 해석
배치는 기준년월(BASE_YM) 단위로 TB_CARD_SALES_LEDGER에서 취소되지 않은(Y) 사용 가능한 거래를 추출하고, 가맹점 마스터 테이블(TB_BOOK_PERF_MERCHANT, TB_TRAD_MARKET_MERCHANT, TB_GENERAL_DEDUCT_MERCHANT)과 LEFT JOIN하여 가맹점 유형을 판별한다. 각 유형별 금액과 건수를 집계한 후 TB_DEDUCTION_MONTHLY_SUMMARY 테이블에 INSERT하고, 동일 기준년월의 기존 데이터를 DELETE하여 Delete Insert 방식으로 재처리한다. job.py는 DB 접속 및 파라미터 처리 로직을 포함하며, test_job.py는 파일 존재 여부와 SQL/파라미터 검증만 수행한다.

## 검증 항목
| 항목 | 결과 | 상세 |
|---|---|---|
| batch_spec.json | PASS | 생성 파일이 존재합니다. |
| query.sql | PASS | 생성 파일이 존재합니다. |
| job.py | PASS | 생성 파일이 존재합니다. |
| SQL 위험 패턴 | PASS | 명백한 위험 SQL 패턴은 발견되지 않았습니다. |
| SQL 파라미터와 batch_spec 일치성 | PASS | SQL 파라미터가 batch_spec.parameters와 연결됩니다: base_ym |
| spec 테이블과 SQL 일치성 | PASS | batch_spec의 테이블 후보가 SQL에서 확인됩니다. |
| 요청 목적 적합성 | PASS | 고객별 월별 소득공제 대상 이용금액 및 가맹점 유형별 금액/건수 집계 목적을 SQL의 SELECT SUM/COUNT와 GROUP BY로 충족 |
| SQL 의미 일치성 | PASS | batch_spec의 source(TB_CARD_SALES_LEDGER), target(TB_DEDUCTION_MONTHLY_SUMMARY), parameters(base_ym)가 SQL의 WHERE 및 INSERT에 반영됨 |
| 파라미터 일치성 | PASS | job.py 실행 단서에서 --base-ym 파라미터가 SQL의 :base_ym 바인드 변수와 일치 |
| 파일 출력 설정 | WARN | batch_spec의 output_format, output_file_pattern, output_dir, encoding이 null로 설정되어 있어 파일 출력 배치인지 테이블 적재 배치인지 명확하지 않음 |
| 운영 재처리 위험 | WARN | DELETE FROM TB_DEDUCTION_MONTHLY_SUMMARY WHERE BASE_YM = :base_ym 후 INSERT로 멱등성 보장 필요, 재처리 시 중복 적재 위험 존재 |
| 성능 위험 | WARN | TB_CARD_SALES_LEDGER의 SALES_DT, MERCHANT_ID, CANCEL_YN 컬럼에 인덱스 필요, 대량 데이터 조회 시 Full Scan 가능성 높음 |
| 데이터 품질 검증 | WARN | NOT NULL 컬럼(CUSTOMER_ID, BASE_YM)은 검증되나, MERCHANT_TYPE의 'UNKNOWN' 처리 및 금액 합계/건수 정확성 검증 로직 부재 |
| 테스트 충분성 | WARN | test_job.py는 파일 존재 여부와 SQL/파라미터 기본 검증만 수행하며, 집계 정확성, 중복 방지, 성능 테스트 등 심화 검증 미흡 |

## 경고
- output_format, output_file_pattern, output_dir, encoding이 null로 설정되어 있어 파일 출력 배치인지 테이블 적재 배치인지 명확하지 않음
- DELETE 후 INSERT 방식으로 멱등성 보장 필요, 재처리 시 중복 적재 위험 존재
- TB_CARD_SALES_LEDGER의 SALES_DT, MERCHANT_ID, CANCEL_YN 컬럼에 인덱스 필요, 대량 데이터 조회 시 Full Scan 가능성 높음
- MERCHANT_TYPE의 'UNKNOWN' 처리 및 금액 합계/건수 정확성 검증 로직 부재
- test_job.py는 파일 존재 여부와 SQL/파라미터 기본 검증만 수행하며, 집계 정확성, 중복 방지, 성능 테스트 등 심화 검증 미흡

## 점수 산정 근거
```json
{
  "policy_version": "practical-scoring-v3-nonblocking-warn",
  "final_score": 0.82,
  "rule_score": 1.0,
  "llm_score": 0.75,
  "valid_policy": "실행 차단 FAIL만 blocking. 테스트/성능/품질/재처리 보완은 WARN. blocking 없으면 PASS_WITH_WARNINGS",
  "blocking_fail_checks": [],
  "downgraded_fail_checks": [],
  "score_policy": {
    "policy_version": "practical-scoring-v3-nonblocking-warn",
    "effective_rule_score": 1.0,
    "effective_llm_score": 0.75,
    "base_score_before_penalty": 0.863,
    "warn_penalty": 0.02,
    "risk_penalty": 0.055,
    "pass_count": 9,
    "warn_count": 5,
    "fail_count_after_normalization": 0,
    "has_blocking_fail": false,
    "score_policy": "실행 차단 오류만 FAIL. 테스트/성능/품질/재처리 보완은 WARN. blocking 없으면 PASS_WITH_WARNINGS 영역 유지"
  },
  "risk_penalty": {
    "total_penalty": 0.055,
    "penalties": {
      "join_complexity": 0.018000000000000002,
      "aggregation_group_by": 0.01,
      "aggregate_functions": 0.012,
      "case_classification": 0.006,
      "table_load": 0.006,
      "condition_complexity": 0.006,
      "performance_review": 0.008,
      "reprocess_or_duplication_review": 0.008,
      "data_quality_review": 0.008,
      "warning_volume": 0.0075
    },
    "signals": {
      "join_count": 3,
      "aggregate_count": 2,
      "warning_count": 5,
      "has_group_by": true,
      "has_case": true,
      "has_insert": true,
      "has_delete_or_delete_insert": false,
      "scoring_source": "query.sql for structural risk; checks/warnings only for review signals"
    }
  }
}
```

## 권장사항
- 운영 반영 전 실제 DB 컬럼 존재 여부와 컬럼 타입을 확인하세요.
- 기준일자/기간 조건 컬럼에 적절한 인덱스가 있는지 확인하세요.
- 파일 생성 배치라면 output_dir 권한과 파일명 중복/덮어쓰기 정책을 확인하세요.
- 대량 데이터 기준 row count, not null, 중복 건수 검증을 추가하세요.
- LLM 검증은 보조 검증이므로 최종 승인 기준은 룰 검증과 테스트 결과를 함께 보세요.
- output_format, output_file_pattern, output_dir, encoding을 운영 관점에서 명시적으로 설정하거나 파일 출력 배치인지 테이블 적재 배치인지 명확히 구분
- DELETE 후 INSERT 방식의 멱등성 보장을 위해 트랜잭션 처리 또는 UPSERT 로직 검토
- TB_CARD_SALES_LEDGER에 SALES_DT, MERCHANT_ID, CANCEL_YN 인덱스 추가 검토
- MERCHANT_TYPE 'UNKNOWN' 처리 기준 및 금액 합계/건수 정확성 검증 로직 추가
- test_job.py에 집계 정확성, 중복 방지, 성능 테스트 케이스 추가
