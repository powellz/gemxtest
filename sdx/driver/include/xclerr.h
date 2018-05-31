/**
 * Copyright (C) 2016-2017 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

/**
 * Copyright (C) 2017 Xilinx, Inc
 * Author: Umang Parekh
 * Error status related structs and defines
 *
 * This file is dual licensed.  It may be redistributed and/or modified
 * under the terms of the Apache 2.0 License OR version 2 of the GNU
 * General Public License.
 */

#ifndef XCLERR_H_
#define XCLERR_H_

    enum xclFirewallID {
        XCL_FW_MGMT_CONTROL = 0x00000000,
        XCL_FW_USER_CONTROL = 0x00000001,
        XCL_FW_DATAPATH     = 0x00000002
    };

    struct xclAXIErrorStatus {
      unsigned long       mErrFirewallTime;
      unsigned            mErrFirewallStatus;
      enum xclFirewallID  mErrFirewallID;
    };

    struct xclPCIErrorStatus {
      unsigned mDeviceStatus;
      unsigned mUncorrErrStatus;
      unsigned mCorrErrStatus;
      unsigned rsvd1;
      unsigned rsvd2;
    };


    struct xclErrorStatus {
      unsigned  mNumFirewalls;
      struct xclAXIErrorStatus mAXIErrorStatus[8];
      struct xclPCIErrorStatus mPCIErrorStatus;
    };

#endif /* XCLERR_H_ */
