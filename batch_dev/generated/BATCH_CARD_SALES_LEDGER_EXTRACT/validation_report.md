# 🔍 배치 생성 검증 리포트

- 최종 상태: **✅ PASS WITH WARNINGS**
- 점수: **0.85**
- 배치 유형: **db_to_file**
- 검증정책: **practical-scoring-v3-nonblocking-warn**

## 요약
소득공제 대상 거래 추출 배치가 배치 요청서의 목적과 batch_type에 맞게 설계되었으며, SQL과 job.py 실행 단서가 기본 요구사항을 충족하고 있으나, 성능 및 데이터 품질 검증 측면에서 운영 반영 전 확인이 필요한 위험이 존재합니다. 운영 반영 전 검토사항이 있습니다.

## 배치 해석
이 배치는 TB_CARD_SALES_LEDGER(매출원장)에서 BASE_YM 기준년월과 CANCEL_YN='N', USE_YN='Y' 조건을 만족하는 거래를 추출하고, LEFT JOIN을 통해 TB_BOOK_PERF_MERCHANT, TB_TRAD_MARKET_MERCHANT, TB_GENERAL_DEDUCT_MERCHANT 참조 테이블과 매칭하여 MERCHANT_TYPE을 도서공연/전통시장/일반소득공제로 분류합니다. 결과는 CSV 파일(card_sales_ledger_{base_ym}.csv)로 출력되며, UTF-8-sig 인코딩을 사용합니다. job.py는 DB 접속, SQL 실행, 파라미터 처리, 파일 출력 로직을 포함하고 있습니다.

## 검증 항목
| 항목 | 결과 | 상세 |
|---|---|---|
| batch_spec.json | PASS | 생성 파일이 존재합니다. |
| query.sql | PASS | 생성 파일이 존재합니다. |
| job.py | PASS | 생성 파일이 존재합니다. |
| SQL 위험 패턴 | PASS | 명백한 위험 SQL 패턴은 발견되지 않았습니다. |
| SQL 파라미터와 batch_spec 일치성 | PASS | SQL 파라미터가 batch_spec.parameters와 연결됩니다: base_ym |
| 출력 형식 | PASS | 출력 형식이 정의되어 있습니다: csv |
| 출력 파일명 패턴 | PASS | 파일명 패턴이 정의되어 있습니다: card_sales_ledger_{base_ym}.csv |
| spec 테이블과 SQL 일치성 | PASS | batch_spec의 테이블 후보가 SQL에서 확인됩니다. |
| 요청 목적 적합성 | PASS | 배치명 '소득공제 대상 거래 추출 배치'와 업무 목적 '소득공제 대상 거래 추출 및 가맹점 유형별 생성'이 일치하며, BASE_YM 기준년월을 기준으로 월 배치로 설계되었습니다. |
| SQL 의미 일치성 | PASS | FROM TB_CARD_SALES_LEDGER, LEFT JOIN 3개 참조 테이블, WHERE 조건으로 CANCEL_YN='N', USE_YN='Y', BASE_YM=:base_ym, SALES_DT 기간 매칭, MERCHANT_TYPE CASE 분류가 배치 요청서의 처리 내용과 정확히 일치합니다. |
| 파라미터 일치성 | PASS | batch_spec의 parameters에 base_ym(required)이 정의되어 있고, SQL에서 :base_ym으로 사용되며, job.py 실행 단서에서 파라미터 처리 로직이 포함되어 있습니다. |
| 파일 출력 설정 | PASS | output_format='csv', output_file_pattern='card_sales_ledger_{base_ym}.csv', output_dir='./output', encoding='utf-8-sig'가 운영 관점에서 적절한 설정으로 확인됩니다. |
| 운영 재처리 위험 | WARN | 파일 덮어쓰기 방식으로 보이며, 중복 적재 방지를 위한 파일명 중복 체크나 삭제 후 적재 로직이 job.py에 명시되어 있지 않아 멱등성 확인이 필요합니다. |
| 성능 위험 | WARN | TB_CARD_SALES_LEDGER 전체 테이블 조회 후 3개 참조 테이블과 LEFT JOIN 수행으로 Full Scan 가능성이 높으며, SALES_DT, MERCHANT_ID, BASE_YM 컬럼에 인덱스가 필요합니다. |
| 데이터 품질 검증 | WARN | NOT NULL 컬럼 검증은 있으나, MERCHANT_TYPE이 'UNKNOWN'인 경우 처리 로직, 중복 거래 검출, 금액 합계 검증 로직이 SQL에 포함되어 있지 않아 추가 검증이 필요합니다. |
| 테스트 충분성 | PASS | test_job.py에서 batch_spec, query.sql, job.py 존재 여부와 SQL 내용 검증을 수행하며, 기본 산출물 존재 및 SQL 구조 검증을 포함하고 있습니다. |

## 경고
- 성능 위험: TB_CARD_SALES_LEDGER 전체 테이블 조회 후 3개 참조 테이블과 LEFT JOIN 수행으로 Full Scan 가능성이 높으며, SALES_DT, MERCHANT_ID, BASE_YM 컬럼에 인덱스가 필요합니다.
- 운영 재처리 위험: 파일 덮어쓰기 방식으로 보이며, 중복 적재 방지를 위한 파일명 중복 체크나 삭제 후 적재 로직이 job.py에 명시되어 있지 않아 멱등성 확인이 필요합니다.
- 데이터 품질 검증: MERCHANT_TYPE이 'UNKNOWN'인 경우 처리 로직, 중복 거래 검출, 금액 합계 검증 로직이 SQL에 포함되어 있지 않아 추가 검증이 필요합니다.

## 점수 산정 근거
```json
{
  "policy_version": "practical-scoring-v3-nonblocking-warn",
  "final_score": 0.85,
  "rule_score": 1.0,
  "llm_score": 0.85,
  "valid_policy": "실행 차단 FAIL만 blocking. 테스트/성능/품질/재처리 보완은 WARN. blocking 없으면 PASS_WITH_WARNINGS",
  "blocking_fail_checks": [],
  "downgraded_fail_checks": [],
  "score_policy": {
    "policy_version": "practical-scoring-v3-nonblocking-warn",
    "effective_rule_score": 1.0,
    "effective_llm_score": 0.85,
    "base_score_before_penalty": 0.917,
    "warn_penalty": 0.012,
    "risk_penalty": 0.055,
    "pass_count": 13,
    "warn_count": 3,
    "fail_count_after_normalization": 0,
    "has_blocking_fail": false,
    "score_policy": "실행 차단 오류만 FAIL. 테스트/성능/품질/재처리 보완은 WARN. blocking 없으면 PASS_WITH_WARNINGS 영역 유지"
  },
  "risk_penalty": {
    "total_penalty": 0.055,
    "penalties": {
      "join_complexity": 0.018000000000000002,
      "case_classification": 0.006,
      "condition_complexity": 0.006,
      "performance_review": 0.008,
      "reprocess_or_duplication_review": 0.008,
      "data_quality_review": 0.008,
      "warning_volume": 0.0045000000000000005
    },
    "signals": {
      "join_count": 3,
      "aggregate_count": 0,
      "warning_count": 3,
      "has_group_by": false,
      "has_case": true,
      "has_insert": false,
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
- SALES_DT, MERCHANT_ID, BASE_YM 컬럼에 인덱스를 추가하여 조회 성능을 개선하세요.
- 파일 생성 시 중복 방지를 위해 파일명 중복 체크 또는 삭제 후 적재 로직을 job.py에 추가하세요.
- MERCHANT_TYPE='UNKNOWN' 거래에 대한 별도 처리 로직 또는 검증 절차를 추가하세요.
- 대량 데이터 처리 시 메모리 사용량을 모니터링하고, 필요시 배치 크기 조정 또는 스트리밍 처리를 고려하세요.
