using System;

namespace CUETools.Codecs
{
    public class AudioSamples
    {
        public const uint UINT32_MAX = 0xffffffff;

        unsafe public static void Interlace(int* res, int* src1, int* src2, int n)
        {
            for (int i = n; i > 0; i--)
            {
                *(res++) = *(src1++);
                *(res++) = *(src2++);
            }
        }

        unsafe public static void Deinterlace(int* dst1, int* dst2, int* src, int n)
        {
            for (int i = n; i > 0; i--)
            {
                *(dst1++) = *(src++);
                *(dst2++) = *(src++);
            }
        }

        unsafe public static bool MemCmp(int* res, int* smp, int n)
        {
            for (int i = n; i > 0; i--)
                if (*(res++) != *(smp++))
                    return true;
            return false;
        }

        unsafe public static void MemCpy(uint* res, uint* smp, int n)
        {
            for (int i = n; i > 0; i--)
                *(res++) = *(smp++);
        }

        unsafe public static void MemCpy(int* res, int* smp, int n)
        {
            for (int i = n; i > 0; i--)
                *(res++) = *(smp++);
        }

        unsafe public static void MemCpy(long* res, long* smp, int n)
        {
            for (int i = n; i > 0; i--)
                *(res++) = *(smp++);
        }

        unsafe public static void MemCpy(short* res, short* smp, int n)
        {
            for (int i = n; i > 0; i--)
                *(res++) = *(smp++);
        }

        unsafe public static void MemCpy(byte* res, byte* smp, int n)
        {
            if ((((IntPtr)smp).ToInt64() & 7) == (((IntPtr)res).ToInt64() & 7) && n > 32)
            {
                int delta = (int)((8 - (((IntPtr)smp).ToInt64() & 7)) & 7);
                for (int i = delta; i > 0; i--)
                    *(res++) = *(smp++);
                n -= delta;

                MemCpy((long*)res, (long*)smp, n >> 3);
                int n8 = (n >> 3) << 3;
                n -= n8;
                smp += n8;
                res += n8;
            }
            if ((((IntPtr)smp).ToInt64() & 3) == (((IntPtr)res).ToInt64() & 3) && n > 16)
            {
                int delta = (int)((4 - (((IntPtr)smp).ToInt64() & 3)) & 3);
                for (int i = delta; i > 0; i--)
                    *(res++) = *(smp++);
                n -= delta;

                MemCpy((int*)res, (int*)smp, n >> 2);
                int n4 = (n >> 2) << 2;
                n -= n4;
                smp += n4;
                res += n4;
            }
            for (int i = n; i > 0; i--)
                *(res++) = *(smp++);
        }

        unsafe public static void MemSet(int* res, int smp, int n)
        {
            for (int i = n; i > 0; i--)
                *(res++) = smp;
        }

        unsafe public static void MemSet(long* res, long smp, int n)
        {
            for (int i = n; i > 0; i--)
                *(res++) = smp;
        }

        unsafe public static void MemSet(byte* res, byte smp, int n)
        {
            if (IntPtr.Size == 8 && (((IntPtr)res).ToInt64() & 7) == 0 && smp == 0 && n > 8)
            {
                MemSet((long*)res, 0, n >> 3);
                int n8 = (n >> 3) << 3;
                n -= n8;
                res += n8;
            }
            if ((((IntPtr)res).ToInt64() & 3) == 0 && smp == 0 && n > 4)
            {
                MemSet((int*)res, 0, n >> 2);
                int n4 = (n >> 2) << 2;
                n -= n4;
                res += n4;
            }
            for (int i = n; i > 0; i--)
                *(res++) = smp;
        }

        unsafe public static void MemSet(byte[] res, byte smp, int offs, int n)
        {
            fixed (byte* pres = &res[offs])
                MemSet(pres, smp, n);
        }

        unsafe public static void MemSet(int[] res, int smp, int offs, int n)
        {
            fixed (int* pres = &res[offs])
                MemSet(pres, smp, n);
        }

        unsafe public static void MemSet(long[] res, long smp, int offs, int n)
        {
            fixed (long* pres = &res[offs])
                MemSet(pres, smp, n);
        }
    }

}
