HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=/opt/rocm
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

SOURCES = vectoradd_hip.cpp
OBJECTS = $(SOURCES:.cpp=.o)
KERNEL = vectoradd_kernel.cpp

EXECUTABLE=./vectoradd_hip.out
CODEOBJ=./vectoradd_kernel.code

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g

CXX=$(HIPCC)

$(CODEOBJ): $(KERNEL)
	$(HIPCC) $(CXXFLAGS) --genco -save-temps $(KERNEL) -o $(CODEOBJ)

$(EXECUTABLE): $(OBJECTS) $(CODEOBJ)
	$(HIPCC) $(CXXFLAGS) $(OBJECTS) -o $@


test: $(EXECUTABLE)
	$(EXECUTABLE)

debug: $(EXECUTABLE)
	$(HIP_PATH)/bin/rocgdb -ex 'break vectoradd_kernel.cpp:17' $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(CODEOBJ)
	rm -f *.bc *.hipi *.s *.resolution.txt
	rm -f $(HIP_PATH)/src/*.o


