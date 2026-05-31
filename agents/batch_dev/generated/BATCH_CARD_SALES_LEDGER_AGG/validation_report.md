# 🔍 생성 결과 검증

- 상태: **PASS WITH WARNINGS**
- 점수: **0.86**
- 배치유형: **aggregation_to_table**

## 요약
TB_CARD_SALES_LEDGER 기준 데이터를 집계하여 결과를 생성하는 배치입니다.

## 주요 처리
- TB_CARD_SALES_LEDGER 조회
- 사용여부/적용기간 조건 필터링
- 집계 결과 생성

## 검토 필요
- USE_YN / APPLY_START_DT 조건 인덱스 확인
- 집계 기준별 row count 및 중복 검증 확인
- 금액 합계 검증 확인
- row count 및 중복 검증 확인
