/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "com_intel_analytics_bigdl_ppml_dcap_Attestation.h"
#include "sgx_dcap_quoteverify.h"
#include "sgx_ql_quote.h"
#include "sgx_urts.h"
#include <assert.h>
#include <fstream>
#include <sgx_uae_launch.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

using namespace std;

void readCollateralContent(sgx_ql_qve_collateral_t* sgx_collateral, const string& filePath) {
    ifstream file(filePath, ios::binary);
    if (!file.is_open()) {
        printf("Error: Unable to open file %s\n", filePath.c_str());
        return;
    }
    file.read((char*)&sgx_collateral->version, sizeof(sgx_collateral->version));
    file.read((char*)&sgx_collateral->tee_type, sizeof(sgx_collateral->tee_type));

    file.read((char*)&sgx_collateral->pck_crl_issuer_chain_size, sizeof(sgx_collateral->pck_crl_issuer_chain_size));
    sgx_collateral->pck_crl_issuer_chain = new char[sgx_collateral->pck_crl_issuer_chain_size];
    file.read(sgx_collateral->pck_crl_issuer_chain, sgx_collateral->pck_crl_issuer_chain_size);

    file.read((char*)&sgx_collateral->root_ca_crl_size, sizeof(sgx_collateral->root_ca_crl_size));
    sgx_collateral->root_ca_crl = new char[sgx_collateral->root_ca_crl_size];
    file.read(sgx_collateral->root_ca_crl, sgx_collateral->root_ca_crl_size);

    file.read((char*)&sgx_collateral->pck_crl_size, sizeof(sgx_collateral->pck_crl_size));
    sgx_collateral->pck_crl = new char[sgx_collateral->pck_crl_size];
    file.read(sgx_collateral->pck_crl, sgx_collateral->pck_crl_size);

    file.read((char*)&sgx_collateral->tcb_info_issuer_chain_size, sizeof(sgx_collateral->tcb_info_issuer_chain_size));
    sgx_collateral->tcb_info_issuer_chain = new char[sgx_collateral->tcb_info_issuer_chain_size];
    file.read(sgx_collateral->tcb_info_issuer_chain, sgx_collateral->tcb_info_issuer_chain_size);

    file.read((char*)&sgx_collateral->tcb_info_size, sizeof(sgx_collateral->tcb_info_size));
    sgx_collateral->tcb_info = new char[sgx_collateral->tcb_info_size];
    file.read(sgx_collateral->tcb_info, sgx_collateral->tcb_info_size);

    file.read((char*)&sgx_collateral->qe_identity_issuer_chain_size, sizeof(sgx_collateral->tcb_info_size));
    sgx_collateral->qe_identity_issuer_chain = new char[sgx_collateral->qe_identity_issuer_chain_size];
    file.read(sgx_collateral->qe_identity_issuer_chain, sgx_collateral->qe_identity_issuer_chain_size);

    file.read((char*)&sgx_collateral->qe_identity_size, sizeof(sgx_collateral->qe_identity_size));
    sgx_collateral->qe_identity = new char[sgx_collateral->qe_identity_size];
    file.read(sgx_collateral->qe_identity, sgx_collateral->qe_identity_size);

    file.close();
}

void free_collateral(sgx_ql_qve_collateral_t* collateral) {
    if (collateral->pck_crl_issuer_chain != NULL) {
        free(collateral->pck_crl_issuer_chain);
        collateral->pck_crl_issuer_chain = NULL;
    }
    if (collateral->root_ca_crl != NULL) {
        free(collateral->root_ca_crl);
        collateral->root_ca_crl = NULL;
    }
    if (collateral->pck_crl != NULL) {
        free(collateral->pck_crl);
        collateral->pck_crl = NULL;
    }
    if (collateral->tcb_info_issuer_chain != NULL) {
        free(collateral->tcb_info_issuer_chain);
        collateral->tcb_info_issuer_chain = NULL;
    }
    if (collateral->tcb_info != NULL) {
        free(collateral->tcb_info);
        collateral->tcb_info = NULL;
    }
    if (collateral->qe_identity_issuer_chain != NULL) {
        free(collateral->qe_identity_issuer_chain);
        collateral->qe_identity_issuer_chain = NULL;
    }
    if (collateral->qe_identity != NULL) {
        free(collateral->qe_identity);
        collateral->qe_identity = NULL;
    }
}

/**
 * @param quote - ECDSA quote buffer
 * @return verification result (int) 0 success, 1 failed, -1 error
 */
int ecdsa_quote_verification(vector<uint8_t> quote, sgx_ql_qve_collateral_t *p_quote_collateral ) {
  int ret = 0;
  time_t current_time = 0;
  uint32_t supplemental_data_size = 0;
  uint8_t *p_supplemental_data = NULL;
  quote3_error_t dcap_ret = SGX_QL_ERROR_UNEXPECTED;
  sgx_ql_qv_result_t quote_verification_result = SGX_QL_QV_RESULT_UNSPECIFIED;
  uint32_t collateral_expiration_status = 1;

  // quote verification
  // call DCAP quote verify library to get supplemental data size
  dcap_ret = sgx_qv_get_quote_supplemental_data_size(&supplemental_data_size);
  if (dcap_ret == SGX_QL_SUCCESS &&
      supplemental_data_size == sizeof(sgx_ql_qv_supplemental_t)) {
    printf("\tInfo: sgx_qv_get_quote_supplemental_data_size successfully "
           "returned.\n");
    p_supplemental_data = (uint8_t *)malloc(supplemental_data_size);
  } else {
    if (dcap_ret != SGX_QL_SUCCESS)
      printf(
          "\tError: sgx_qv_get_quote_supplemental_data_size failed: 0x%04x\n",
          dcap_ret);

    if (supplemental_data_size != sizeof(sgx_ql_qv_supplemental_t))
      printf("\tWarning: sgx_qv_get_quote_supplemental_data_size returned size "
             "is not same with header definition in SGX SDK, please make sure "
             "you are using same version of SGX SDK and DCAP QVL.\n");

    supplemental_data_size = 0;
  }

  // set current time.
  current_time = time(NULL);

  // call DCAP quote verify library for quote verification
  // here you can choose 'trusted' or 'untrusted' quote verification by
  // specifying parameter '&qve_report_info' if '&qve_report_info' is NOT NULL,
  dcap_ret = sgx_qv_verify_quote(quote.data(), (uint32_t)quote.size(), p_quote_collateral,
                                 current_time, &collateral_expiration_status,
                                 &quote_verification_result, NULL,
                                 supplemental_data_size, p_supplemental_data);
  if (dcap_ret == SGX_QL_SUCCESS) {
    printf("\tInfo: sgx_qv_verify_quote successfully returned.\n");
  } else {
    printf("\tError: sgx_qv_verify_quote failed: 0x%04x\n", dcap_ret);
    printf("please refer to P65 of https://download.01.org/intel-sgx/latest/dcap-latest/linux"
        "/docs/Intel_SGX_ECDSA_QuoteLibReference_DCAP_API.pdf for more information\n");
  }

  // check verification result
  switch (quote_verification_result) {
  case SGX_QL_QV_RESULT_OK:
    ret = 0;
    break;
  case SGX_QL_QV_RESULT_CONFIG_NEEDED:
    printf("The SGX platform firmware and SW are at the latest security patching level"
            "but there are platform hardware configurations"
            "that may expose the enclave to vulnerabilities.\n");
    ret = 1;
    break;
  case SGX_QL_QV_RESULT_OUT_OF_DATE:
    printf("The SGX platform firmware and SW are not at the latest security patching level."
            "The platform needs to be patched with firmware and/or software patches.\n");
      ret = 1;
      break;
  case SGX_QL_QV_RESULT_OUT_OF_DATE_CONFIG_NEEDED:
    printf("The SGX platform firmware and SW are not at the latest security patching level."
           "The platform needs to be patched with firmware and/or software patches.\n");
      ret = 1;
      break;
  case SGX_QL_QV_RESULT_SW_HARDENING_NEEDED:
    printf("The SGX platform firmware and SW are at the latest security patching level"
     "but there are certain vulnerabilities that can only be mitigated with"
      "software mitigations implemented by the enclave.\n");
       ret = 1;
       break;
  case SGX_QL_QV_RESULT_CONFIG_AND_SW_HARDENING_NEEDED:
    printf("The SGX platform firmware and SW are at the latest security patching level"
            "but there are certain vulnerabilities that can only be mitigated with"
            "software mitigations implemented by the enclave.\n");
    ret = 1;
    break;
  case SGX_QL_QV_RESULT_INVALID_SIGNATURE:
    printf("\tThe signature over the application report is invalid\n");
      ret = -1;
      break;
  case SGX_QL_QV_RESULT_REVOKED:
    printf("\tThe attestation key or platform has been revoked\n");
      ret = -1;
      break;
  case SGX_QL_QV_RESULT_UNSPECIFIED:
    printf("\tThe Quote verification failed due to an error in one of the input\n");
      ret = -1;
      break;
  default:
    ret = -1;
    break;
  }

  if (ret == 0) {
    printf("\tSuccess: Verification completed successfully.\n");
  } else if (ret == 1) {
    printf("\tWarning: Verification completed with Non-terminal result: %x\n",
            quote_verification_result);
  } else {
    printf("\tError: Verification completed with Terminal result: %x\n",
            quote_verification_result);
  }
  return ret;
}

JNIEXPORT jint JNICALL
Java_com_intel_analytics_bigdl_ppml_dcap_Attestation_sdkVerifyQuote(
    JNIEnv *env, jclass cls, jbyteArray quote) {
  // Return -1 if quote is null
  if (quote == NULL)
    return -1;
  // convert jbyteArray to vector<char>
  jbyte *jbae = env->GetByteArrayElements(quote, 0);
  jsize len = env->GetArrayLength(quote);
  // Copy quote to quote_vector
  char *quote_arrary = (char *)jbae;
  vector<unsigned char> quote_vector;
  for (int i = 0; i < len; i++) {
    quote_vector.push_back(quote_arrary[i]);
  }
  int result = ecdsa_quote_verification(quote_vector, NULL);
  return result;
}

JNIEXPORT jint JNICALL
Java_com_intel_analytics_bigdl_ppml_dcap_Attestation_sdkOfflineVerifyQuote(
    JNIEnv *env, jclass cls, jbyteArray quote, jstring collateralFile) {
  // Return -1 if quote is null
  if (quote == NULL)
    return -1;
  // convert jbyteArray to vector<char>
  jbyte *jbae = env->GetByteArrayElements(quote, 0);
  jsize len = env->GetArrayLength(quote);
  // Copy quote to quote_vector
  char *quote_arrary = (char *)jbae;
  vector<unsigned char> quote_vector;
  for (int i = 0; i < len; i++) {
    quote_vector.push_back(quote_arrary[i]);
  }

  sgx_ql_qve_collateral_t *p_quote_collateral = NULL;
  p_quote_collateral = (sgx_ql_qve_collateral_t *)malloc(sizeof(sgx_ql_qve_collateral_t));

  const char *cstr = env->GetStringUTFChars(collateralFile, nullptr);
  std::string collateral_path(cstr);
  env->ReleaseStringUTFChars(collateralFile, cstr);

  readCollateralContent(p_quote_collateral, collateral_path);

  int result = ecdsa_quote_verification(quote_vector, p_quote_collateral);
  if (p_quote_collateral != NULL) {
        free_collateral(p_quote_collateral);
        free(p_quote_collateral);
  }
  return result;
}
