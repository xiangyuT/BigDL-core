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

package com.intel.analytics.bigdl.ppml.tdx;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class AttestationTest {

    @Test
    public void isAttestationLoaded() throws Exception {
        assertTrue(Attestation.isAttestationLoaded());
    }

    @Test
    public void tdxGenerateQuoteTest() throws Exception {
        Attestation tdx = new Attestation();
        String res = new String(tdx.tdxGenerateQuote());
        assertTrue(res != "");
    }
}