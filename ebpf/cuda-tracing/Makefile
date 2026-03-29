# SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)

USER_TARGETS   := cuda_trace
BPF_TARGETS    := cuda_trace.bpf
BPF_SKEL_OBJ := cuda_trace.bpf.o

# CFLAGS := -D__TARGET_ARCH_x86
# LDLIBS     += -lnvidia-ml
NVML_DIR= //usr/lib/x86_64-linux-gnu/
ifneq ($(wildcard $(NVML_DIR)/libnvidia-ml.so),)
  LDLIBS += -L$(NVML_DIR) -Wl,-rpath,$(NVML_DIR) -lnvidia-ml
else ifneq ($(wildcard $(NVML_DIR)/libnvidia-ml.so.1),)
  LDLIBS += -L$(NVML_DIR) -Wl,-rpath,$(NVML_DIR) -Wl,--no-as-needed -l:libnvidia-ml.so.1
else
  $(info [cuda-trace] NVML not found; building without NVML)
  CFLAGS += -DNO_NVML
endif
USER_LIBS = -lmnl -lpthread
EXTRA_DEPS += cuda_evt.h

LIB_DIR = ../lib

include $(LIB_DIR)/common.mk
