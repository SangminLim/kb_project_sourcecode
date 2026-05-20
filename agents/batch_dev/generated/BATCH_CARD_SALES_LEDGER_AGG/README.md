# 매출원장테이블 월별 집계

## Batch ID
`BATCH_CARD_SALES_LEDGER_AGG`

## Batch Type
`aggregation_to_table`

## 설명
[배치 개발 요청서] 배치명: 소득공제 월별 통합 집계 배치 기준 테이블: TB_CARD_SALES_LEDGER 참조 테이블: TB_BOOK_PERF_MERCHANT TB_TRAD_MARKET_MERCHANT TB_GENERAL_DEDUCT_MERCHANT target_table: TB_DEDUCTION_MONTHLY_SUMMARY 출력 목적: 고객별 월별 소득공제 대상 이용금액을 집계한다. 기준: BASE_YM 처리 내용: 매출 거래 원장을 기준으로 소득공제 대상 거래를 판별한다. 가맹점 분류 마스터를 참조하여 가맹점 유형을 구분한다. SALES_DT 기준으로 APPLY_START_DT ~ APPLY_END_DT 유효기간을 매칭한다. 취소 거래는 제외한다. 고객별 / 기준년월 / 가맹점유형 기준으로 금액과 건수를 집계한다. Delete Insert 방식으로 적재한다. 조인 조건: TB_CARD_SALES_LEDGER.MERCHANT_ID = TB_BOOK_PERF_MERCHANT.MERCHANT_ID TB_CARD_SALES_LEDGER.MERCHANT_ID = TB_TRAD_MARKET_MERCHANT.MERCHANT_ID TB_CARD_SALES_LEDGER.MERCHANT_ID = TB_GENERAL_DEDUCT_MERCHANT.MERCHANT_ID 가맹점 유형: TB_BOOK_PERF_MERCHANT 매칭 시 BOOK_PERF TB_TRAD_MARKET_MERCHANT 매칭 시 TRAD_MARKET TB_GENERAL_DEDUCT_MERCHANT 매칭 시 GENERAL 조건: CANCEL_YN = 'N' USE_YN = 'Y' SALES_DT BETWEEN APPLY_START_DT AND APPLY_END_DT 집계 기준: CUSTOMER_ID BASE_YM MERCHANT_TYPE 집계 항목: TOTAL_AMT = SUM(SALES_AMT) TXN_COUNT = COUNT(*) 출력 컬럼: CUSTOMER_ID BASE_YM MERCHANT_TYPE TOTAL_AMT TXN_COUNT 배치 유형: 월별 집계 배치 Delete Insert 방식

## 실행 예시

```bash
python job.py --database-url "$DATABASE_URL" --base-ym 202604
```

## 처리 방식
- 기준년월(`base_ym`) 단위로 대상 데이터를 삭제한다.
- ERWin 메타 기반으로 생성된 집계 INSERT SQL을 실행한다.
- 결과는 파일로 생성하지 않고 대상 테이블에 적재한다.

## 재수행 방식
- `delete_sql`로 기준년월 범위 삭제
- `query.sql`의 INSERT SQL 재실행

## 검토 필요사항
- ERWin 메타와 실제 DB 컬럼 일치 여부 확인
- JOIN 결과 중복 여부 확인
- 기준년월 재수행 시 DELETE 범위 확인
- 집계 금액 및 건수 검증
- 실행 계획 및 인덱스 확인
