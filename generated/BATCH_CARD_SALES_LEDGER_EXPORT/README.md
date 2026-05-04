# 소득공제 월별 통합 집계 배치 기준: - 기준년월(BASE_YM) 대상

## Batch ID
`BATCH_CARD_SALES_LEDGER_EXPORT`

## Batch Type
`db_to_file`

## 설명
[배치 개발 요청서] 배치명: 소득공제 월별 통합 집계 배치 기준: - 기준년월(BASE_YM) 대상 테이블: - TB_CARD_SALES_LEDGER - TB_BOOK_PERF_MERCHANT - TB_TRAD_MARKET_MERCHANT - TB_GENERAL_DEDUCT_MERCHANT 처리 내용: - 매출원장 기준으로 가맹점별 소득공제 대상 거래를 구분한다 - 도서공연 / 전통시장 / 일반 가맹점을 각각 분류한다 - 유효기간(APPLY_START_DT ~ APPLY_END_DT) 기준으로 가맹점 매칭한다 - 취소 거래 제외 - 고객별, 가맹점 유형별 월별 금액 집계 출력: - 고객ID, 기준년월, 가맹점유형, 총금액, 건수 조건: - CANCEL_YN = 'N' - USE_YN = 'Y'

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
