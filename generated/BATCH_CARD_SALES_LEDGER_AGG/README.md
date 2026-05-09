# 매출원장테이블 월별 집계

## Batch ID
`BATCH_CARD_SALES_LEDGER_AGG`

## Batch Type
`aggregation_to_table`

## 설명
소득공제 월별 통합 집계 배치 만들어줘

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
