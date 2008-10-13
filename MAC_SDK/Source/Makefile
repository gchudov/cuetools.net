#
# Simple Makefile
#
# Frank Klemm
#

TARGET   = mac
INCLUDES = -IShared -IMACLib -IConsole
CPPOPT   = -s -O3 -Wall -pedantic -D__GNUC_IA32__
COMPILER = gcc

SOURCEFILES = \
Console/Console.cpp		\
MACLib/APECompress.cpp		\
MACLib/APECompressCore.cpp	\
MACLib/APECompressCreate.cpp	\
MACLib/APEDecompress.cpp	\
MACLib/APEInfo.cpp		\
MACLib/APELink.cpp		\
MACLib/APESimple.cpp		\
MACLib/APETag.cpp		\
MACLib/BitArray.cpp		\
MACLib/MACLib.cpp		\
MACLib/MACProgressHelper.cpp	\
MACLib/NNFilter.cpp		\
MACLib/NewPredictor.cpp		\
MACLib/Prepare.cpp		\
MACLib/UnBitArray.cpp		\
MACLib/UnBitArrayBase.cpp	\
MACLib/WAVInputSource.cpp	\
Shared/GlobalFunctions.cpp	\
Shared/StdLibFileIO.cpp		\
Shared/WinFileIO.cpp		\
MACLib/NNFilterAsm.o



$(TARGET): $(SOURCEFILES)
	$(COMPILER) -static $(CPPOPT) $(INCLUDES) -o $(TARGET)-static $(SOURCEFILES)
	$(COMPILER)         $(CPPOPT) $(INCLUDES) -o $(TARGET)        $(SOURCEFILES)

MACLib/NNFilterAsm.o : MACLib/NNFilterAsm.nas
	nasm -f elf -o MACLib/NNFilterAsm.o MACLib/NNFilterAsm.nas -l MACLib/NNFilterAsm.lst

APE_Source.tar.bz2:
	@sh ./MakeSourceBall

test:
	@echo e4dd45d9b5ec4cc91f3bd2210a543df6
	@./$(TARGET) Adagio.ape - -d | md5sum

speed:
	@sync
	@cat Adagio.ape > /dev/null
	@sync
	time ./mac Adagio.ape /dev/null -d


# Samual Barber: Adagio for Strings (10:10.84)
#
# C version		203.01 sec
# First ASM version      76.89 sec
#                        76.85 sec
#			 77.12 sec
#                        76.23 sec
#			 75.67 sec
#                        76.26 sec
#                        76.79 sec
#                        76.70 sec
