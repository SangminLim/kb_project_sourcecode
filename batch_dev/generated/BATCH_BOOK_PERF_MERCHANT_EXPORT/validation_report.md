# 🔍 배치 생성 검증 리포트

- 최종 상태: **✅ PASS WITH WARNINGS**
- 점수: **0.88**
- 배치 유형: **db_to_file**
- 검증정책: **practical-scoring-v3-nonblocking-warn**

## 요약
도서공연 가맹점 파일 생성 배치가 정상적으로 생성되었으며, SQL과 job.py 실행 단서가 배치 목적에 부합하고 운영 관점에서 적절한 설정을 갖추고 있다. 운영 반영 전 검토사항이 있습니다.

## 배치 해석
이 배치는 TB_BOOK_PERF_MERCHANT 테이블에서 소득공제 대상 도서공연 가맹점 정보를 추출하여 CSV 파일로 출력하는 일 배치이다. 기준일자(base_date)는 APPLY_START_DT와 APPLY_END_DT 사이의 데이터만 조회하고, USE_YN이 'Y'인 레코드만 필터링한다. SQL은 동적 파라미터(:base_date)를 사용하여 기준일자 조건을 처리하며, APPLY_END_DT가 NULL인 경우를 OR 조건으로 처리하여 기간 범위 조회 로직을 완성했다. job.py는 DB 접속, SQL 실행, CSV 파일 생성 로직을 포함하고 있으며, README와 test_job.py는 배치 산출물 존재 여부를 검증하는 테스트를 제공한다.

## 검증 항목
| 항목 | 결과 | 상세 |
|---|---|---|
| batch_spec.json | PASS | 생성 파일이 존재합니다. |
| query.sql | PASS | 생성 파일이 존재합니다. |
| job.py | PASS | 생성 파일이 존재합니다. |
| SQL 위험 패턴 | PASS | 명백한 위험 SQL 패턴은 발견되지 않았습니다. |
| SQL 파라미터와 batch_spec 일치성 | PASS | SQL 파라미터가 batch_spec.parameters와 연결됩니다: base_date |
| 출력 형식 | PASS | 출력 형식이 정의되어 있습니다: csv |
| 출력 파일명 패턴 | PASS | 파일명 패턴이 정의되어 있습니다: book_perf_merchant_{base_date}.csv |
| spec 테이블과 SQL 일치성 | PASS | batch_spec의 테이블 후보가 SQL에서 확인됩니다. |
| 요청 목적 적합성 | PASS | 배치명 '도서공연 가맹점 파일 생성'과 출력 목적 '소득공제 대상 도서공연 가맹점 정보 추출'이 일치하며, 출력 형식 CSV와 파일명 패턴 book_perf_merchant_{base_date}.csv가 요청과 동일하다. |
| SQL 의미 일치성 | PASS | FROM TB_BOOK_PERF_MERCHANT, WHERE USE_YN = 'Y', 기준일자 BETWEEN APPLY_START_DT AND APPLY_END_DT 조건이 배치 요청과 정확히 매핑되었다. |
| 파라미터 일치성 | PASS | batch_spec의 parameters에 base_date(YYYYMMDD)가 필수 파라미터로 정의되었고, SQL에서 :base_date로 사용되었다. |
| 파일 출력 설정 | PASS | output_format이 csv, output_file_pattern이 book_perf_merchant_{base_date}.csv, output_dir이 ./output, encoding이 utf-8-sig로 운영 관점에서 적절한 설정이다. |
| 운영 재처리 위험 | WARN | 파일 덮어쓰기 방지를 위한 파일명 중복 체크 로직이 명시되지 않았으며, 멱등성 보장을 위한 추가 검증이 필요하다. |
| 성능 위험 | WARN | TB_BOOK_PERF_MERCHANT 테이블에 APPLY_START_DT, APPLY_END_DT, USE_YN 컬럼에 대한 인덱스 존재 여부가 확인되지 않아 대량 데이터 조회 시 Full Scan 위험이 있다. |
| 테스트 충분성 | WARN | test_job.py는 배치 산출물 존재 여부만 검증하며, SQL/파일포맷/파라미터 유효성 검증 로직이 포함되어 있지 않다. |

## 경고
- 파일 덮어쓰기 방지를 위한 파일명 중복 체크 로직이 명시되지 않음
- APPLY_START_DT, APPLY_END_DT, USE_YN 컬럼에 대한 인덱스 존재 여부 미확인으로 대량 데이터 조회 시 성능 저하 가능성
- test_job.py가 SQL/파일포맷/파라미터 유효성 검증까지 수행하지 않음

## 점수 산정 근거
```json
{
  "policy_version": "practical-scoring-v3-nonblocking-warn",
  "final_score": 0.88,
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
    "risk_penalty": 0.025,
    "pass_count": 12,
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
      "warning_volume": 0.0045000000000000005
    },
    "signals": {
      "join_count": 0,
      "aggregate_count": 0,
      "warning_count": 3,
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
- 파일명 중복 체크 및 멱등성 보장을 위한 로직 추가
- APPLY_START_DT, APPLY_END_DT, USE_YN 컬럼에 인덱스 적용 여부 확인 및 필요 시 인덱스 추가
- test_job.py에 SQL 실행 결과 검증, 파일 포맷 검증, 파라미터 유효성 검증 로직 추가
