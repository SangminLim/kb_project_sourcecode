# 도서공연가맹점테이블 파일 생성

## Batch ID
`BATCH_BOOK_PERF_MERCHANT_EXPORT`

## Batch Type
`db_to_file`

## 설명
도서공연 가맹점 파일 생성 배치 만들어줘

## 실행 예시

```bash
python job.py --database-url "$DATABASE_URL" --base-date 20260428 --output-dir ./output
```

## 출력 파일
`book_perf_merchant_{base_date}.csv`

## 검토 필요사항
- query.sql의 테이블/컬럼/조건이 실제 운영 기준과 맞는지 확인
- 인덱스 사용 여부와 실행 계획 확인
- 파일 구분자, 인코딩, 헤더 포함 여부 확인
- 건수/NULL/중복/금액 합계 검증 조건 추가
