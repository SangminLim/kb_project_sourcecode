# 소득공제 대상 거래 추출 배치

## Batch ID
`BATCH_CARD_SALES_LEDGER_EXTRACT`

## Batch Type
`db_to_file`

## 설명
[배치 개발 요청서] 배치명: 소득공제 대상 거래 추출 배치 기준 테이블: - TB_CARD_SALES_LEDGER 참조 테이블: - TB_BOOK_PERF_MERCHANT - TB_TRAD_MARKET_MERCHANT - TB_GENERAL_DEDUCT_MERCHANT 업무 목적: - 소득공제 대상 거래를 추출한다. - 가맹점 유형별 소득공제 대상 거래를 생성한다. 기준: - BASE_YM 처리 내용: - 매출원장 테이블 기준으로 거래 데이터를 조회한다. - 가맹점ID 기준으로 가맹점 분류 마스터와 JOIN한다. - 거래일자(SALES_DT)와 가맹점 적용기간(APPLY_START_DT ~ APPLY_END_DT)을 비교하여 유효한 가맹점만 매칭한다. - 취소거래는 제외한다. - 사용여부가 'Y'인 가맹점만 처리한다. - 도서공연 / 전통시장 / 일반소득공제 가맹점 유형을 구분한다. 출력 컬럼: - SALES_SEQ_NO - SALES_DT - CUSTOMER_ID - MERCHANT_ID - SALES_AMT - BASE_YM - MERCHANT_TYPE 조건: - CANCEL_YN = 'N' - USE_YN = 'Y' - SALES_DT BETWEEN APPLY_START_DT AND APPLY_END_DT 배치 유형: - 대상 거래 추출 배치 - 월 배치

## 처리 방식
- 거래 원장 테이블 기준으로 대상 거래를 조회한다.
- ERWIN relation 정보를 기준으로 classification_master 테이블과 JOIN한다.
- 거래일자와 마스터 적용기간을 비교한다.
- 취소 거래를 제외한다.
- 매칭된 classification_value를 MERCHANT_TYPE으로 출력한다.

## 실행 예시

```bash
python job.py --database-url "$DATABASE_URL" --base-ym 202604 --output-dir ./output
```

## 출력 파일
`card_sales_ledger_{base_ym}.csv`

## 검토 필요사항
- ERWIN relation 기준 JOIN 조건 검토
- 기간 조건 검토
- 중복 매칭 여부 검토
- 파일 인코딩 및 구분자 검토
