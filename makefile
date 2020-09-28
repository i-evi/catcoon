CC = gcc

# Build Ctrl Flag
BCTRL =

INC   += -I ./src/
LINK  += -lm

# Debug flag, cc_assert
DFLAG += -DENABLE_CC_ASSERT
# AddressSanitizer for gcc/clang
DFLAG += -g -fsanitize=address -fno-omit-frame-pointer

CFLAG += # -std=c89
CFLAG += -Wall # -Wpedantic

OFLAG += # -O3

# Enable OpenMP
OFLAG += -DENABLE_OPENMP -fopenmp

CFLAG += $(DFLAG) $(WFLAG) $(OFLAG)

# Enable automatic tensor manager
CFLAG += -DAUTO_TSRMGR

# Configurations for utilities
UTIM_COND   =
UT_LIST_CFG = -DENABLE_FOPS

# 3rd party source/library configurations
# stb: Read/write jpg, png, tga format images.
# Will only support bmp image if disabled
3RDSRC_CF += stb

# parg: Argv parser written in ANSI C
# Argv parser for apps
3RDSRC_CF += parg

# seb: sequential encoded buffers
# Pack and save compressed models files
3RDSRC_CF += seb

# zlib: General-purpose compression and decompression library
# Pack and save compressed models files
3RDSRC_CF += zlib

# cJSON: Ultralightweight JSON parser in ANSI C.
3RDSRC_CF += cjson

ifneq ($(findstring MINI, $(BCTRL)),)
	CC = tcc
	CFLAG  = -std=c89
	CFLAG += -DAUTO_TSRMGR
	3RDSRC_CF = parg
endif

ifneq ($(findstring -std=c89, $(CFLAG)),)
	CFLAG += -DCONFIG_STD_C89
	ifneq ($(findstring $(CC), gcc clang),)
		WFLAG += -Wno-long-long
	endif
endif

ifneq ($(findstring stb, $(3RDSRC_CF)),)
	UTIM_COND += -DUSE3RD_STB_IMAGE -I ./src/3rd_party/stb/
endif

ifneq ($(findstring parg, $(3RDSRC_CF)),)
	ALL_O   += parg.o
	APP_INC += -I ./src/3rd_party/parg/
endif

ifneq ($(findstring seb, $(3RDSRC_CF)),)
	ALL_O += seb.o fastlz.o
	UT_LIST_CFG += -DENABLE_SEB -I ./src/3rd_party/seb/
endif

ifneq ($(findstring zlib, $(3RDSRC_CF)),)
	LINK += -lz
	UT_LIST_CFG += -DENABLE_ZLIB
endif

ifneq ($(findstring cjson, $(3RDSRC_CF)),)
	ALL_O   += cJSON.o
	APP_INC += -I ./src/3rd_party/cjson/
endif

ALL_O += \
catcoon.o cc_tensor.o cc_dtype.o cc_tsrmgr.o cc_fmap2d.o cc_pool2d.o \
cc_array.o cc_basic.o cc_actfn.o cc_fullycon.o cc_pad2d.o cc_cpufn.o \
cc_conv2d.o cc_dsc2d.o cc_normfn.o cc_image.o util_rbt.o util_list.o \
util_log.o util_vec.o util_image.o global_fn_cfg.o 

CATCOON_A = libcatcoon.a

APPS_DEMO = simple lenet lenet_pack lenet_unpack 
APPS_UTIL = packager listpkg

APP_NAMES  = $(APPS_DEMO) $(APPS_UTIL)
APP_INC   += $(INC)
APP_LINK  += $(LINK)

ifeq ($(OS),Windows_NT)
	RM = del
	APPS = $(foreach v, $(APP_NAMES), $(v).exe)
else
	RM = rm
	APPS = $(APP_NAMES)
endif

all: $(APPS) # $(CATCOON_A)

%.o: ./src/%.c
	$(CC) -c -o $@ $< $(CFLAG) $(INC)

# Apps For Linux
%: ./demo/%.c $(ALL_O)
	$(CC) -o $@ $< $(ALL_O) $(CFLAG) $(APP_INC) $(APP_LINK)
%: ./util/%.c $(ALL_O)
	$(CC) -o $@ $< $(ALL_O) $(CFLAG) $(APP_INC) $(APP_LINK)
# Apps For Windows
%.exe: ./demo/%.c $(ALL_O)
	$(CC) -o $@ $< $(ALL_O) $(CFLAG) $(APP_INC) $(APP_LINK)
%.exe: ./util/%.c $(ALL_O)
	$(CC) -o $@ $< $(ALL_O) $(CFLAG) $(APP_INC) $(APP_LINK)

global_fn_cfg.o : $(patsubst %, ./src/%, global_fn_cfg.h global_fn_cfg.c)

catcoon.o     : $(patsubst %, ./src/%, catcoon.h catcoon.c)
cc_actfn.o    : $(patsubst %, ./src/%, cc_actfn.h cc_actfn.c)
cc_array.o    : $(patsubst %, ./src/%, cc_array.h cc_array.c)
cc_basic.o    : $(patsubst %, ./src/%, cc_basic.h cc_basic.c)
cc_conv2d.o   : $(patsubst %, ./src/%, cc_conv2d.h cc_conv2d.c)
cc_dsc2d.o    : $(patsubst %, ./src/%, cc_dsc2d.h cc_dsc2d.c)
cc_cpufn.o    : $(patsubst %, ./src/%, cc_cpufn.h cc_cpufn.c)
cc_dtype.o    : $(patsubst %, ./src/%, cc_dtype.h cc_dtype.c)
cc_fmap2d.o   : $(patsubst %, ./src/%, cc_fmap2d.h cc_fmap2d.c)
cc_fullycon.o : $(patsubst %, ./src/%, cc_fullycon.h cc_fullycon.c)
cc_image.o    : $(patsubst %, ./src/%, cc_image.h cc_image.c)
cc_normfn.o   : $(patsubst %, ./src/%, cc_normfn.h cc_normfn.c)
cc_pad2d.o    : $(patsubst %, ./src/%, cc_pad2d.h cc_pad2d.c)
cc_pool2d.o   : $(patsubst %, ./src/%, cc_pool2d.h cc_pool2d.c)
cc_tsrmgr.o   : $(patsubst %, ./src/%, cc_tsrmgr.h cc_tsrmgr.c)
cc_tensor.o   : $(patsubst %, ./src/%, cc_tensor.h cc_tensor.c)

util_log.o   : $(patsubst %, ./src/%, util_log.h util_log.c)
util_vec.o   : $(patsubst %, ./src/%, util_vec.h util_vec.c)
util_rbt.o   : $(patsubst %, ./src/%, util_rbt.h util_rbt.c)
util_list.o  : $(patsubst %, ./src/%, util_list.h util_list.c)
	$(CC) -c -o $@ ./src/util_list.c $(CFLAG) $(UT_LIST_CFG)
util_image.o : $(patsubst %, ./src/%, util_image.h util_image.c)
	$(CC) -c -o $@ ./src/util_image.c $(CFLAG) $(UTIM_COND)

# 3rd party objs
parg.o: ./src/3rd_party/parg/parg*
	$(CC) -c -o $@ ./src/3rd_party/parg/parg.c $(CFLAG)

seb.o : ./src/3rd_party/seb/seb*
	$(CC) -c -o $@ ./src/3rd_party/seb/seb.c $(CFLAG)
fastlz.o : ./src/3rd_party/seb/fastlz*
	$(CC) -c -o $@ ./src/3rd_party/seb/fastlz.c $(CFLAG)

cJSON.o : ./src/3rd_party/cjson/cJSON*
	$(CC) -c -o $@ ./src/3rd_party/cjson/cJSON.c $(CFLAG)

minimal:
	$(MAKE) "BCTRL = MINI"

clean:
	$(RM) *.o && $(RM) $(APPS)
