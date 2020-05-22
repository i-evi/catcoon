cc = gcc

# Debug flag, cc_assert
DFLAG += -DENABLE_CC_ASSERT
# AddressSanitizer for gcc/clang
DFLAG += # -g -fsanitize=address -fno-omit-frame-pointer

CFLAG += # -std=c89
CFLAG += -Wall # -Wpedantic

OFLAG += -Os

# Enable OpenMP
OFLAG += -DENABLE_OPENMP -fopenmp

CFLAG += $(DFLAG) $(WFLAG) $(OFLAG)

# Enable automatic tensor manager
CFLAG += -DAUTO_TSRMGR

# 3rd party source/library configurations
# stb for jpg, png, tga format images,
# will only support bmp image if disabled
3RDSRC_CF += stb

ifneq ($(findstring -std=c89, $(CFLAG)),)
	CFLAG += -DCONFIG_STD_C89
	ifneq ($(findstring $(cc), gcc clang),)
		WFLAG += -Wno-long-long
	endif
endif

LINK += -lm
INC  += -I ./src/
ALL_O = catcoon.o cc_tensor.o cc_dtype.o cc_image.o cc_array.o cc_tsrmgr.o \
	cc_basic.o cc_fmap2d.o cc_pad2d.o cc_conv2d.o cc_pool2d.o cc_cpufn.o \
	cc_fullycon.o global_fn_cfg.o util_rbt.o util_list.o util_log.o \
	util_image.o

APP_NAMES = simple lenet

ifeq ($(OS),Windows_NT)
	RM = del
	APPS =  $(foreach v, $(APP_NAMES), $(v).exe)
else
	RM = rm
	APPS = $(APP_NAMES)
endif

all: $(APPS)

%.o: ./src/%.c
	$(cc) -c -o $@ $< $(CFLAG) $(INC)
# APPS For Linux
%: ./demo/%.c $(ALL_O)
	$(cc) -o $@ $< $(ALL_O) $(CFLAG) $(INC) $(LINK)
# APPS For Windows
%.exe: ./demo/%.c $(ALL_O)
	$(cc) -o $@ $< $(ALL_O) $(CFLAG) $(INC) $(LINK)

global_fn_cfg.o : $(patsubst %, ./src/%, global_fn_cfg.h global_fn_cfg.c)

catcoon.o     : $(patsubst %, ./src/%, catcoon.h catcoon.c)
cc_array.o    : $(patsubst %, ./src/%, cc_array.h cc_array.c)
cc_basic.o    : $(patsubst %, ./src/%, cc_basic.h cc_basic.c)
cc_conv2d.o   : $(patsubst %, ./src/%, cc_conv2d.h cc_conv2d.c)
cc_cpufn.o    : $(patsubst %, ./src/%, cc_cpufn.h cc_cpufn.c)
cc_dtype.o    : $(patsubst %, ./src/%, cc_dtype.h cc_dtype.c)
cc_fmap2d.o   : $(patsubst %, ./src/%, cc_fmap2d.h cc_fmap2d.c)
cc_fullycon.o : $(patsubst %, ./src/%, cc_fullycon.h cc_fullycon.c)
cc_image.o    : $(patsubst %, ./src/%, cc_image.h cc_image.c)
cc_pad2d.o    : $(patsubst %, ./src/%, cc_pad2d.h cc_pad2d.c)
cc_pool2d.o   : $(patsubst %, ./src/%, cc_pool2d.h cc_pool2d.c)
cc_tsrmgr.o   : $(patsubst %, ./src/%, cc_tsrmgr.h cc_tsrmgr.c)
cc_tensor.o   : $(patsubst %, ./src/%, cc_tensor.h cc_tensor.c)

util_log.o   : $(patsubst %, ./src/%, util_log.h util_log.c)
util_rbt.o   : $(patsubst %, ./src/%, util_rbt.h util_rbt.c)
util_list.o  :  ./src/util_list.h ./src/util_list.c
	$(cc) -c -o $@ ./src/util_list.c $(CFLAG) -DENABLE_FOPS
UTIM_COND = 
ifneq ($(findstring stb, $(3RDSRC_CF)),)
	UTIM_COND += -DUSE3RD_STB_IMAGE -I ./src/3rd_party/stb/
endif
util_image.o : ./src/util_image.h ./src/util_image.c
	$(cc) -c -o $@ ./src/util_image.c $(CFLAG) $(UTIM_COND)

clean:
	$(RM) *.o && $(RM) $(APPS)