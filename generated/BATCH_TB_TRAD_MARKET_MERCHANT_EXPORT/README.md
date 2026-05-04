# BATCH_TB_TRAD_MARKET_MERCHANT_EXPORT

## 목적
[배치 개발 요청서] 배치명: 전통시장 가맹점 파일 생성 기준 테이블: TB_TRAD_MARKET_MERCHANT 출력 목적: 소득공제 대상 전통시장 가맹점 추출 출력 형식: CSV 파일명: traditional_market_merchant_YYYYMMDD.csv 기준일자: APPLY_START_DT 조건: - 기준일자가 적용시작일자와 종료일자 사이 - USE_YN = 'Y'

## 실행 예시

```bash
python generated/BATCH_TB_TRAD_MARKET_MERCHANT_EXPORT/job.py \
  --database-url "mysql+pymysql://user:pass@localhost:3306/testDB?charset=utf8mb4" \
  --base-ym 202604
```

## 생성 SQL
```sql
SELECT
    MERCHANT_ID,
    MERCHANT_NM,
    MARKET_NM,
    BIZ_NO,
    APPLY_START_DT,
    APPLY_END_DT,
    USE_YN,
    REG_DTM,
    UPD_DTM
FROM TB_TRAD_MARKET_MERCHANT
WHERE :base_date BETWEEN APPLY_START_DT AND IFNULL(APPLY_END_DT, '99991231')
```

## 운영 반영 전 검토
- ERWin/실제 DB 컬럼 일치 여부
- 조인 결과 중복 여부
- 기준년월 재수행 시 delete_insert 범위
- 집계 금액 검증
