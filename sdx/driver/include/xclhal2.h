/**
 * Copyright (C) 2015-2017, Xilinx Inc - All rights reserved
 * Xilinx SDAccel HAL userspace driver APIs
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

#ifndef _XCL_HAL2_H_
#define _XCL_HAL2_H_

#ifdef __cplusplus
#include <cstdlib>
#include <cstdint>
#else
#include <stdlib.h>
#include <stdint.h>
#endif

#if defined(_WIN32)
#ifdef XCL_DRIVER_DLL_EXPORT
#define XCL_DRIVER_DLLESPEC __declspec(dllexport)
#else
#define XCL_DRIVER_DLLESPEC __declspec(dllimport)
#endif
#else
#define XCL_DRIVER_DLLESPEC __attribute__((visibility("default")))
#endif


#include "xclperf.h"
#include "xcl_app_debug.h"
#include "xclerr.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef void * xclDeviceHandle;

    struct xclBin;
    struct axlf;
    /**
     * Structure used to obtain various bits of information from the device.
     */

    struct xclDeviceInfo2 {
      unsigned mMagic; // = 0X586C0C6C; XL OpenCL X->58(ASCII), L->6C(ASCII), O->0 C->C L->6C(ASCII);
      char mName[256];
      unsigned short mHALMajorVersion;
      unsigned short mHALMinorVersion;
      unsigned short mVendorId;
      unsigned short mDeviceId;
      unsigned short mSubsystemId;
      unsigned short mSubsystemVendorId;
      unsigned short mDeviceVersion;
      size_t mDDRSize;                    // Size of DDR memory
      size_t mDataAlignment;              // Minimum data alignment requirement for host buffers
      size_t mDDRFreeSize;                // Total unused/available DDR memory
      size_t mMinTransferSize;            // Minimum DMA buffer size
      unsigned short mDDRBankCount;
      unsigned short mOCLFrequency[4];
      unsigned short mPCIeLinkWidth;
      unsigned short mPCIeLinkSpeed;
      unsigned short mDMAThreads;
      short mOnChipTemp;
      short mFanTemp;
      unsigned short  mVInt;
      unsigned short  mVAux;
      unsigned short  mVBram;
      float mCurrent;
      unsigned short mNumClocks;
      unsigned short mFanSpeed;
      bool mMigCalib;
      // More properties here
    };

    /**
     * xclMemoryDomains is for support of legacy APIs
     * It is not used in BO APIs where we instead use xclBOKind
     */
    enum xclMemoryDomains {
        XCL_MEM_HOST_RAM =    0x00000000,
        XCL_MEM_DEVICE_RAM =  0x00000001,
        XCL_MEM_DEVICE_BRAM = 0x00000002,
        XCL_MEM_SVM =         0x00000003,
        XCL_MEM_CMA =         0x00000004,
        XCL_MEM_DEVICE_REG  = 0x00000005
    };

    /* byte-0 lower 4 bits for DDR Flags are one-hot encoded */
    enum xclDDRFlags {
        XCL_DEVICE_RAM_BANK0 = 0x00000000,
        XCL_DEVICE_RAM_BANK1 = 0x00000002,
        XCL_DEVICE_RAM_BANK2 = 0x00000004,
        XCL_DEVICE_RAM_BANK3 = 0x00000008
    };

    /**
     * Defines Buffer Object Kind which represents a fragment of device accesible memory
     * and the corresponding backing host memory.

     * 1. Shared virtual memory (SVM) class of systems like CAPI or MPSoc with SMMU. BOs
     *    have a common host RAM backing store.
     *    XCL_BO_SHARED_VIRTUAL
     *
     * 2. Shared physical memory class of systems like Zynq (or MPSoc with pass though SMMU)
     *    with Linux CMA buffer allocation. BOs have common host CMA allocated backing store.
     *    XCL_BO_SHARED_PHYSICAL
     *
     * 3. Shared virtual memory (SVM) class of systems with dedicated RAM and device MMU. BOs
     *    have a device RAM dedicated backing store and another host RAM allocated backing store.
     *    The buffers are sync'd via DMA. Both physical buffers use the same virtual address,
     *    hence giving the effect of SVM.
     *    XCL_BO_MIRRORED_VIRTUAL
     *
     * 4. Dedicated memory class of devices like PCIe card with DDR. BOs have a device RAM
     *    dedicated backing store and another host RAM allocated backing store. The buffers
     *    are sync'd via DMA
     *    XCL_BO_DEVICE_RAM     *
     *
     * 5. Dedicated onchip memory class of devices like PCIe card with BRAM. BOs have a device
     *    BRAM dedicated backing store and another host RAM allocated backing store. The buffers
     *    are sync'd via DMA
     *    XCL_BO_DEVICE_BRAM
     */

    enum xclBOKind {
        XCL_BO_SHARED_VIRTUAL = 0,
        XCL_BO_SHARED_PHYSICAL,
        XCL_BO_MIRRORED_VIRTUAL,
        XCL_BO_DEVICE_RAM,
        XCL_BO_DEVICE_BRAM,
        XCL_BO_DEVICE_PREALLOCATED_BRAM,
    };

    enum xclBOSyncDirection {
        XCL_BO_SYNC_BO_TO_DEVICE = 0,
        XCL_BO_SYNC_BO_FROM_DEVICE,
    };

    /**
     * Define address spaces on the device AXI bus. The enums are used in xclRead() and xclWrite()
     * to pass relative offsets.
     */

    enum xclAddressSpace {
        XCL_ADDR_SPACE_DEVICE_FLAT = 0,     // Absolute address space
        XCL_ADDR_SPACE_DEVICE_RAM = 1,      // Address space for the DDR memory
        XCL_ADDR_KERNEL_CTRL = 2,           // Address space for the OCL Region control port
        XCL_ADDR_SPACE_DEVICE_PERFMON = 3,  // Address space for the Performance monitors
        XCL_ADDR_SPACE_DEVICE_CHECKER = 5,  // Address space for protocol checker
        XCL_ADDR_SPACE_MAX = 8
    };

    /**
     * Defines verbosity levels which are passed to xclOpen during device creation time
     */

    enum xclVerbosityLevel {
        XCL_QUIET = 0,
        XCL_INFO = 1,
        XCL_WARN = 2,
        XCL_ERROR = 3
    };

    enum xclResetKind {
        XCL_RESET_KERNEL,
        XCL_RESET_FULL
    };

    struct xclDeviceUsage {
        size_t h2c[8];
        size_t c2h[8];
        size_t ddrMemUsed[8];
        unsigned ddrBOAllocated[8];
        unsigned totalContexts;
        uint64_t xclbinId[4];
    };

    /**
     * @defgroup devman DEVICE MANAGMENT APIs
     * --------------------------------------
     * APIs to open, close, query and program the device
     * @{
     */

    /**
     * Return a count of devices found in the system
     */
    XCL_DRIVER_DLLESPEC unsigned xclProbe();

    /**
     * Open a device and obtain its handle.
     * "deviceIndex" is 0 for first device, 1 for the second device and so on
     * "logFileName" is optional and if not NULL should be used to log messages
     * "level" specifies the verbosity level for the messages being logged to logFileName
     */
    XCL_DRIVER_DLLESPEC xclDeviceHandle xclOpen(unsigned deviceIndex, const char *logFileName,
                                                xclVerbosityLevel level);

    /**
     * Close an opened device
     */
    XCL_DRIVER_DLLESPEC void xclClose(xclDeviceHandle handle);

    /**
     * Reset the device. All running kernels will be killed and buffers in DDR will be
     * purged. A device would be reset if a user's application dies without waiting for
     * running kernel(s) to finish.
     */
    XCL_DRIVER_DLLESPEC int xclResetDevice(xclDeviceHandle handle, xclResetKind kind);

    /**
     * Obtain various bits of information from the device
     */
    XCL_DRIVER_DLLESPEC int xclGetDeviceInfo2(xclDeviceHandle handle, xclDeviceInfo2 *info);

    /**
     * Obtain usage information from the device
     */
    XCL_DRIVER_DLLESPEC int xclGetUsageInfo(xclDeviceHandle handle, xclDeviceUsage *info);

    /**
     * Obtain Error information from the device
     */
    XCL_DRIVER_DLLESPEC int xclGetErrorStatus(xclDeviceHandle handle, xclErrorStatus *info);

    /**
     * Download bitstream to the device. The bitstream is encapsulated inside xclBin.
     * The bitstream may be PR bistream for devices which support PR and full bitstream
     * for devices which require full configuration.
     */
    XCL_DRIVER_DLLESPEC int xclLoadXclBin(xclDeviceHandle handle, const xclBin *buffer);

    /**
     * Set the OCL region frequncy
     */
    //XCL_DRIVER_DLLESPEC int xclReClock(xclDeviceHandle handle, unsigned targetFreqMHz);

    /**
     * Set the OCL region frequncies
     */
    XCL_DRIVER_DLLESPEC int xclReClock2(xclDeviceHandle handle, unsigned short region,
                                        const unsigned short *targetFreqMHz);

    /**
     * Get exclusive ownership of the device. The lock is necessary before performing buffer
     * migration, register access or bitstream downloads.
     */
    XCL_DRIVER_DLLESPEC int xclLockDevice(xclDeviceHandle handle);

    /**
     * Release exclusive ownership of the device obtained via xclLockDevice.
     */
    XCL_DRIVER_DLLESPEC int xclUnlockDevice(xclDeviceHandle handle);

    /**
     * Update the device BPI PROM with new image
     */
    XCL_DRIVER_DLLESPEC int xclUpgradeFirmware(xclDeviceHandle handle, const char *fileName);

    /**
     * Update the device PROM with new image with clearing bitstream
     */
    XCL_DRIVER_DLLESPEC int xclUpgradeFirmware2(xclDeviceHandle handle, const char *file1, const char* file2);

    /**
     * Update the device SPI PROM with new image
     */
    XCL_DRIVER_DLLESPEC int xclUpgradeFirmwareXSpi(xclDeviceHandle handle, const char *fileName, int index);

    /**
     * Boot the FPGA from PROM. This should only be called when there are no other clients
     * It may break the PCIe link and render the device unusable till a reboot of the host
     */
    XCL_DRIVER_DLLESPEC int xclBootFPGA(xclDeviceHandle handle);

    /**
     * Write to /sys/bus/pci/devices/<deviceHandle>/remove and initiate a pci rescan by
     * writing to /sys/bus/pci/rescan.
     */
    XCL_DRIVER_DLLESPEC int xclRemoveAndScanFPGA();

    /**
     * Get the version number. 1 => Hal1 ; 2 => Hal2
     */
    XCL_DRIVER_DLLESPEC unsigned int xclVersion();

    /*
     * Get the physical address on the device
     */
    XCL_DRIVER_DLLESPEC uint64_t xclGetDeviceAddr(xclDeviceHandle handle, unsigned int boHandle);

    /** @} */

    /**
     * @defgroup bufman BUFFER MANAGMENT APIs
     * --------------------------------------
     *
     * Buffer management APIs are used for managing device memory. The board vendors are expected
     * to provide Linux GEM style memory manager with the following set of APIs to perform common
     * operations: alloc, free, pread, pwrite, map, unmap, import, export. The APIs are blocking.
     * @{
     */

    /**
     * Allocate a BO of requested size with appropriate flags. Returns BO Handle.
     *
     * @return
     *   Local handle to the BO for use by this device
     */
    XCL_DRIVER_DLLESPEC unsigned int xclAllocBO(xclDeviceHandle handle, size_t size, xclBOKind domain,
                                                unsigned flags);

    /**
     * Allocate a BO of requested size with user provided host buffer pointer
     */
    XCL_DRIVER_DLLESPEC unsigned int xclAllocUserPtrBO(xclDeviceHandle handle, void *userptr, size_t size,
                                                       unsigned flags);

    /**
     * Free a previously allocated BO
     */
    XCL_DRIVER_DLLESPEC void xclFreeBO(xclDeviceHandle handle, unsigned int boHandle);

    /**
     * Copy host buffer contents to previously allocated device memory. "seek" specifies how many bytes
     * to skip at the beginning of the BO before copying-in "size" bytes of host buffer.
     */
    XCL_DRIVER_DLLESPEC size_t xclWriteBO(xclDeviceHandle handle, unsigned int boHandle,
                                           const void *src, size_t size, size_t seek);

    /**
     * Copy contents of previously allocated device memory to host buffer. "skip" specifies how many bytes
     * to skip from the beginning of the BO before copying-out "size" bytes of device buffer.
     */
    XCL_DRIVER_DLLESPEC size_t xclReadBO(xclDeviceHandle handle, unsigned int boHandle,
                                         void *dst, size_t size, size_t skip);

    /**
     * Map the contents of the buffer object into host memory
     * To unmap the buffer call posix unmap on mapped void * returned from xclMapBO
     */
    XCL_DRIVER_DLLESPEC void *xclMapBO(xclDeviceHandle handle, unsigned int boHandle, bool write);

    XCL_DRIVER_DLLESPEC int xclSyncBO(xclDeviceHandle handle, unsigned int boHandle, xclBOSyncDirection dir,
                                      size_t size, size_t offset);

    /**
     * Export a BO for import into another device.
     *
     * This operation is backed by Linux DMA-BUF framework
     *
     * @param handle
     *   Handle to device
     * @param boHandle
     *   Handle to BO owned by this device which needs to be exported
     * @return
     *   Global file handle to the BO for import by another device
     */
    XCL_DRIVER_DLLESPEC int xclExportBO(xclDeviceHandle handle, unsigned int boHandle);

    /**
     * Import a BO from another device.
     *
     * This operation is backed by Linux DMA-BUF framework
     *
     * @param handle
     *   Handle to device
     * @param boForeignHandle
     *   Global file handle to foreign BO owned by another device which needs to be imported
     * @return
     *   Local handle to the BO for use by this device
     */
    XCL_DRIVER_DLLESPEC unsigned int xclImportBO(xclDeviceHandle handle, int fd, unsigned flags);

    /**
     * Get size info of the BO from BOH
     *
     * This operation is backed by Linux DMA-BUF framework
     *
     * @param handle
     *   Handle to device
     * @return
     *   size_t size of the BO
     */
    XCL_DRIVER_DLLESPEC size_t xclGetBOSize(xclDeviceHandle handle, unsigned int boHandle);


    /** @} */

    /**
     * @defgroup depbufman Deprecated BUFFER MANAGMENT APIs
     * ----------------------------------------------------
     *
     * Do NOT develop new features using the following 5 API's. These are for backwards
     * compatibility with classic HAL interface and are deprecated. New clients should
     * use BO based APIs defined above.
     *
     * @{
     */

    /**
     * Allocate a buffer on the device DDR and return its address
     */
    XCL_DRIVER_DLLESPEC uint64_t xclAllocDeviceBuffer(xclDeviceHandle handle, size_t size);

    /**
     * Allocate a buffer on the device DDR bank and return its address
     */
    XCL_DRIVER_DLLESPEC uint64_t xclAllocDeviceBuffer2(xclDeviceHandle handle, size_t size,
                                                       xclMemoryDomains domain,
                                                       unsigned flags);

    /**
     * Free a previously allocated buffer on the device DDR
     */
    XCL_DRIVER_DLLESPEC void xclFreeDeviceBuffer(xclDeviceHandle handle, uint64_t buf);

    /**
     * Copy host buffer contents to previously allocated device memory. "seek" specifies how many bytes to skip
     * at the beginning of the destination before copying "size" bytes of host buffer.
     */
    XCL_DRIVER_DLLESPEC size_t xclCopyBufferHost2Device(xclDeviceHandle handle, uint64_t dest,
                                                        const void *src, size_t size, size_t seek);

    /**
     * Copy contents of previously allocated device memory to host buffer. "skip" specifies how many bytes to skip
     * from the beginning of the source before copying "size" bytes of device buffer.
     */
    XCL_DRIVER_DLLESPEC size_t xclCopyBufferDevice2Host(xclDeviceHandle handle, void *dest,
                                                        uint64_t src, size_t size, size_t skip);

    /** @} */


    /**
     * @defgroup unmanageddma UNMANGED DMA APIs
     * ----------------------------------------
     *
     * Unmanaged DMA APIs are for exclusive use by the debuggers and tools. The API
     * interface may change in future releases.
     * @{
     */

    /**
     * Write to device address at a specific offset. flags is used to specify address segment.
     *
     * @return
     *   Number of bytes written or error code
     */
    XCL_DRIVER_DLLESPEC ssize_t xclUnmgdPread(xclDeviceHandle handle, unsigned flags, void *buf,
                                              size_t count, uint64_t offset);

    /**
     * Read from device address at a specific offset. flags is used to specify address segment.
     *
     * @return
     *   Number of bytes read or error code
     */
    XCL_DRIVER_DLLESPEC ssize_t xclUnmgdPwrite(xclDeviceHandle handle, unsigned flags, const void *buf,
                                               size_t count, uint64_t offset);

    /** @} */

    /**
     * @defgroup readwrite DEVICE READ AND WRITE APIs
     * ----------------------------------------------
     *
     * These functions are used to read and write peripherals sitting on the address map.  OpenCL runtime will be using the BUFFER MANAGEMNT
     * APIs described above to manage OpenCL buffers. It would use xclRead/xclWrite to program and manage
     * peripherals on the card. For programming the Kernel, OpenCL runtime uses the kernel control register
     * map generated by the OpenCL compiler.
     * Note that the offset is wrt the address space
     * @{
     */
    XCL_DRIVER_DLLESPEC size_t xclWrite(xclDeviceHandle handle, xclAddressSpace space, uint64_t offset,
                                        const void *hostBuf, size_t size);

    XCL_DRIVER_DLLESPEC size_t xclRead(xclDeviceHandle handle, xclAddressSpace space, uint64_t offset,
                                       void *hostbuf, size_t size);

    /**
     * TODO:
     * Define the following APIs
     *
     * 1. Host accessible pipe APIs: pread/pwrite
     * 2. Accelerator status, start, stop APIs
     * 3. Context creation APIs to support multiple clients
     * 4. Multiple OCL Region support
     * 5. DPDK style buffer management and device polling
     *
     */
    /** @} */

    /**
     * @defgroup exec COMPUTE UNIT START/WAIT APIs
     * -------------------------------------------
     *
     * These functions are used to start compute units and wait for them to finish.
     * @{
     */

    /**
     * Submit an exec buffer for execution
     * The function may return with -EAGAIN if command queue is full (all execute slots
     * are full)
     */
    XCL_DRIVER_DLLESPEC int xclExecBuf(xclDeviceHandle handle, unsigned int cmdBO);

    /**
     * Wait for notification from the hardware. The function essentially calls "poll" system
     * call on the driver file handle. The return value has same semantics as poll system call.
     * If return value is > 0 caller should check the status of submitted exec buffers
     */
    XCL_DRIVER_DLLESPEC int xclExecWait(xclDeviceHandle handle, int timeoutMilliSec);

    /**
     * Support for non managed interrupts (interrupts from custom IPs). fd should be obtained from
     * eventfd system call. Caller should use standard poll/read eventfd framework inroder to wait for
     * interrupts. Note this is different than Compute Unit interrupts managed by the core runtime
     * and exposed by xclExecWait()
     */
    XCL_DRIVER_DLLESPEC int xclRegisterInterruptNotify(xclDeviceHandle handle, unsigned int userInterrupt, int fd);

    /** @} */

    /**
     * @defgroup perfmon PERFORMANCE MONITORING OPERATIONS
     * ---------------------------------------------------
     *
     * These functions are used to read and write to the performance monitoring infrastructure.
     * OpenCL runtime will be using the BUFFER MANAGEMNT APIs described above to manage OpenCL buffers.
     * It would use these functions to initialize and sample the performance monitoring on the card.
     * Note that the offset is wrt the address space
     */

    /* Write host event to device tracing (Zynq only) */
    XCL_DRIVER_DLLESPEC void xclWriteHostEvent(xclDeviceHandle handle, xclPerfMonEventType type,
                                               xclPerfMonEventID id);

    XCL_DRIVER_DLLESPEC size_t xclGetDeviceTimestamp(xclDeviceHandle handle);

    XCL_DRIVER_DLLESPEC double xclGetDeviceClockFreqMHz(xclDeviceHandle handle);

    XCL_DRIVER_DLLESPEC double xclGetReadMaxBandwidthMBps(xclDeviceHandle handle);

    XCL_DRIVER_DLLESPEC double xclGetWriteMaxBandwidthMBps(xclDeviceHandle handle);

    XCL_DRIVER_DLLESPEC void xclSetProfilingNumberSlots(xclDeviceHandle handle, xclPerfMonType type,
    		                                            uint32_t numSlots);

    XCL_DRIVER_DLLESPEC uint32_t xclGetProfilingNumberSlots(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC void xclGetProfilingSlotName(xclDeviceHandle handle, xclPerfMonType type,
                                                     uint32_t slotnum, char* slotName, uint32_t length);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonClockTraining(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonStartCounters(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonStopCounters(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonReadCounters(xclDeviceHandle handle, xclPerfMonType type,
    		                                          xclCounterResults& counterResults);

    XCL_DRIVER_DLLESPEC size_t xclDebugReadIPStatus(xclDeviceHandle handle, xclDebugReadType type,
                                                                               void* debugResults);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonStartTrace(xclDeviceHandle handle, xclPerfMonType type,
    		                                        uint32_t startTrigger);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonStopTrace(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC uint32_t xclPerfMonGetTraceCount(xclDeviceHandle handle, xclPerfMonType type);

    XCL_DRIVER_DLLESPEC size_t xclPerfMonReadTrace(xclDeviceHandle handle, xclPerfMonType type,
    		                                       xclTraceResultsVector& traceVector);

    /** @} */


    /**
     * @defgroup Internal OPERATIONS
     * ---------------------------------------------------
     *
     * These functions are for internal use only. External clients should not be using these API's.
     * These are bound to change and Xilinx won't be responsible for maintaining backwards compatibility
     * for these functions.
     *
     */
    XCL_DRIVER_DLLESPEC void intScanDevices();

    /** @} */
    XCL_DRIVER_DLLESPEC int xclXbsak(int argc, char *argv[]);

#ifdef __cplusplus
}
#endif

#endif
