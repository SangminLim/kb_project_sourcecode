# 🔍 배치 생성 검증 리포트

- 최종 상태: **✅ PASS WITH WARNINGS**
- 점수: **0.88**
- 배치 유형: **db_to_file**
- 검증정책: **practical-scoring-v3-nonblocking-warn**

## 요약
일반 소득공제 가맹점 파일 생성 배치가 CSV 출력 형식으로 정상 생성되었으며, SQL과 job.py 실행 단서가 배치 목적과 일치함 운영 반영 전 검토사항이 있습니다.

## 배치 해석
이 배치는 TB_GENERAL_DEDUCT_MERCHANT 테이블에서 일반 소득공제 대상 가맹점 정보를 추출하여 CSV 파일로 출력하는 일 배치입니다. 기준일자(base_date)가 APPLY_START_DT와 APPLY_END_DT 사이에 있고 USE_YN='Y'인 레코드만 조회하며, 출력 컬럼은 MERCHANT_ID, MERCHANT_NM, BIZ_NO, MCC_CD, APPLY_START_DT, APPLY_END_DT, USE_YN입니다. 배치 유형은 db_to_file이며, job.py는 DB 접속, 파라미터 처리, CSV 파일 출력 로직을 포함하고 있습니다.

## 검증 항목
| 항목 | 결과 | 상세 |
|---|---|---|
| batch_spec.json | PASS | 생성 파일이 존재합니다. |
| query.sql | PASS | 생성 파일이 존재합니다. |
| job.py | PASS | 생성 파일이 존재합니다. |
| SQL 위험 패턴 | PASS | 명백한 위험 SQL 패턴은 발견되지 않았습니다. |
| SQL 파라미터와 batch_spec 일치성 | PASS | SQL 파라미터가 batch_spec.parameters와 연결됩니다: base_date |
| 출력 형식 | PASS | 출력 형식이 정의되어 있습니다: csv |
| 출력 파일명 패턴 | PASS | 파일명 패턴이 정의되어 있습니다: general_deduct_merchant_{base_date}.csv |
| spec 테이블과 SQL 일치성 | PASS | batch_spec의 테이블 후보가 SQL에서 확인됩니다. |
| 요청 목적 적합성 | PASS | 배치명, 출력 목적, 출력 형식, 출력 컬럼, 조건이 원본 요청과 일치함 |
| SQL 의미 일치성 | PASS | FROM TB_GENERAL_DEDUCT_MERCHANT, WHERE USE_YN='Y', 기준일자 BETWEEN APPLY_START_DT AND APPLY_END_DT 조건이 배치 목적과 정확히 매핑됨 |
| 파라미터 일치성 | PASS | batch_spec의 parameters(name: base_date)와 SQL의 :base_date 파라미터가 일치하며, resolved_columns에서 base_date가 APPLY_START_DT로 매핑됨 |
| 파일 출력 설정 | PASS | output_format: csv, output_file_pattern: general_deduct_merchant_{base_date}.csv, output_dir: ./output, encoding: utf-8-sig가 운영 관점에서 적절함 |
| job.py 실행 단서 | PASS | CSV 출력 로직, DB 접속/SQL 실행 로직, 파라미터 처리 단서가 존재함 |
| 테스트 충분성 | WARN | test_job.py는 파일 존재 여부만 검증하며, SQL/파일포맷/파라미터 검증 로직이 없음 |
| 성능 위험 | WARN | APPLY_START_DT, APPLY_END_DT, USE_YN 컬럼에 인덱스가 없을 경우 Full Scan 가능성 있음 |
| 데이터 품질 검증 | WARN | NOT NULL 컬럼(MERCHANT_ID, APPLY_START_DT) 검증 로직이 없으며, 중복/row count 검증 부재 |

## 경고
- 테스트 파일이 생성 산출물 존재 여부만 검증하므로 SQL/파일포맷/파라미터 검증 로직 추가 필요
- APPLY_START_DT, APPLY_END_DT, USE_YN 컬럼에 인덱스 확인 필요
- NOT NULL 컬럼(MERCHANT_ID, APPLY_START_DT) 검증 로직 부재
- 중복 레코드 및 row count 검증 로직 부재

## 점수 산정 근거
```json
{
  "policy_version": "practical-scoring-v3-nonblocking-warn",
  "final_score": 0.878,
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
    "warn_penalty": 0.014,
    "risk_penalty": 0.025,
    "pass_count": 13,
    "warn_count": 3,
    "fail_count_after_normalization": 0,
    "has_blocking_fail": false,
    "score_policy": "실행 차단 오류만 FAIL. 테스트/성능/품질/재처리 보완은 WARN. blocking 없으면 PASS_WITH_WARNINGS 영역 유지"
  },
  "risk_penalty": {
    "total_penalty": 0.025,
    "penalties": {
      "condition_complexity": 0.006,
      "performance_review": 0.008,
      "test_coverage_gap": 0.008,
      "reprocess_or_duplication_review": 0.008,
      "data_quality_review": 0.008,
      "warning_volume": 0.006
    },
    "signals": {
      "join_count": 0,
      "aggregate_count": 0,
      "warning_count": 4,
      "has_group_by": false,
      "has_case": false,
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
- test_job.py에 SQL 구문 검증, 파일 포맷 검증, 파라미터 검증 로직 추가
- APPLY_START_DT, APPLY_END_DT, USE_YN 컬럼에 인덱스 적용 여부 확인
- NOT NULL 컬럼 검증 및 중복 레코드 제거 로직 추가
- 대량 데이터 조회 시 성능 테스트 수행
