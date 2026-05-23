# 🔍 생성 결과 검증

- 상태: **PASS WITH WARNINGS**
- 점수: **0.88**
- 배치유형: **db_to_file**

## 요약
TB_GENERAL_DEDUCT_MERCHANT에서 사용가능(USE_YN='Y') + 적용기간 유효 + 기준일자 기준 데이터를 조회하여 CSV 파일로 생성하는 배치입니다.

## 주요 처리
- TB_GENERAL_DEDUCT_MERCHANT 조회
- 기준일자(base_date) 기준 유효 데이터 필터링
- CSV 파일 생성

## 검토 필요
- USE_YN / APPLY_START_DT 조건 인덱스 확인
- CSV 파일 중복 생성 방지 확인
- 출력 헤더/구분자 확인
- 기준일자(base_date) 파라미터 검증 확인
