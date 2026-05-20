# 매출원장테이블 파일 생성

## Batch ID
`BATCH_CARD_SALES_LEDGER_EXPORT`

## Batch Type
`db_to_file`

## 설명
[배치 개발 요청서] 배치명: 소득공제 대상 거래 추출 배치 기준 테이블: TB_CARD_SALES_LEDGER 참조 테이블: TB_BOOK_PERF_MERCHANT TB_TRAD_MARKET_MERCHANT TB_GENERAL_DEDUCT_MERCHANT 출력 목적: 소득공제 대상 거래를 추출한다. 기준: BASE_YM 처리 내용: 매출원장 테이블 기준으로 거래 데이터를 조회한다. 가맹점ID 기준으로 가맹점 분류 마스터와 JOIN한다. 거래일자(SALES_DT)와 가맹점 적용기간(APPLY_START_DT ~ APPLY_END_DT)을 비교하여 유효한 가맹점만 매칭한다. 취소거래는 제외한다. 사용여부가 'Y'인 가맹점만 처리한다. 도서공연 / 전통시장 / 일반소득공제 가맹점 유형을 구분한다. 조건: CANCEL_YN = 'N' USE_YN = 'Y' SALES_DT BETWEEN APPLY_START_DT AND APPLY_END_DT 출력 컬럼: SALES_SEQ_NO SALES_DT CUSTOMER_ID MERCHANT_ID SALES_AMT BASE_YM MERCHANT_TYPE 배치 유형: ledger_extract_with_classification 실행 주기: 월배치

## 실행 예시

```bash
python job.py --database-url "$DATABASE_URL" --base-date 20260428 --output-dir ./output
```

## 출력 파일
`card_sales_ledger_{base_date}.csv`

## 검토 필요사항
- query.sql의 테이블/컬럼/조건이 실제 운영 기준과 맞는지 확인
- 인덱스 사용 여부와 실행 계획 확인
- 파일 구분자, 인코딩, 헤더 포함 여부 확인
- 건수/NULL/중복/금액 합계 검증 조건 추가
