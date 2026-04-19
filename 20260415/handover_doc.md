# 인수인계 업무 문서

## 1. A은행 소득공제 업무
### 업무 설명
A은행에서 발생한 카드 승인 및 매입 데이터, 가맹점 정보, 고객 정보를 수집하여 소득공제 대상 여부를 판별하고 결과를 반영하는 업무

### 배치 작업
#### 1단계: 데이터 적재
- BATCH_01_CARD_APPROVAL_LOAD
- BATCH_02_CARD_PURCHASE_LOAD
- BATCH_03_CUSTOMER_INFO_LOAD
- BATCH_04_MERCHANT_INFO_LOAD
- BATCH_05_CANCEL_DATA_LOAD

#### 2단계: 소득공제 처리
- BATCH_06_DEDUCTION_TARGET_FILTER
- BATCH_07_DEDUCTION_AGGREGATION

#### 3단계: 결과 생성
- BATCH_08_DEDUCTION_FILE_GENERATION

---

## 2. B증권 소득공제 업무
- 계좌 기반 거래 포함
- 금융상품 제외 처리 존재

---

## 3. A은행 청구 업무
### 배치 흐름
- 데이터 적재
- 청구 대상 처리
- 청구서 생성

---

## 4. 개발환경
- Java 1.8
- Eclipse
- Tomcat
- DB Tool

---

## 5. 장애 대응
1. 로그 확인
2. 오류 분석
3. 재처리 판단

---

## 6. 핵심 요약
- 소득공제: 공제 대상 판별 및 집계
- 청구: 금액 계산 및 청구서 생성
