using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs
{
	public class Crc32
	{
		uint[] table = new uint[256];

		public uint ComputeChecksum(uint crc, byte val)
		{
			return (crc >> 8) ^ table[(crc & 0xff) ^ val];
		}

		public uint ComputeChecksum(uint crc, byte[] bytes, int pos, int count)
		{
			for (int i = pos; i < pos + count; i++)
				crc = ComputeChecksum(crc, bytes[i]);
			return crc;
		}

		public uint ComputeChecksum(uint crc, uint s)
		{
			return ComputeChecksum(ComputeChecksum(ComputeChecksum(ComputeChecksum(
				crc, (byte)s), (byte)(s >> 8)), (byte)(s >> 16)), (byte)(s >> 24));
		}

		public unsafe uint ComputeChecksum(uint crc, int * samples, uint count)
		{
			for (uint i = 0; i < count; i++)
			{
				int s1 = samples[2 * i], s2 = samples[2 * i + 1];
				crc = ComputeChecksum(ComputeChecksum(ComputeChecksum(ComputeChecksum(
					crc, (byte)s1), (byte)(s1 >> 8)), (byte)s2), (byte)(s2 >> 8));
			}
			return crc;
		}

		public unsafe uint ComputeChecksumWONULL(uint crc, int* samples, uint count)
		{
			for (uint i = 0; i < count; i++)
			{
				int s1 = samples[2 * i], s2 = samples[2 * i + 1];
				if (s1 != 0)
					crc = ComputeChecksum(ComputeChecksum(crc, (byte)s1), (byte)(s1 >> 8));
				if (s2 != 0)
					crc = ComputeChecksum(ComputeChecksum(crc, (byte)s2), (byte)(s2 >> 8));
			}
			return crc;
		}

		uint Reflect(uint val, int ch)
		{
			uint value = 0;
			// Swap bit 0 for bit 7
			// bit 1 for bit 6, etc.
			for (int i = 1; i < (ch + 1); i++)
			{
				if (0 != (val & 1))
					value |= 1U << (ch - i);
				val >>= 1;
			}
			return value;
		}

		const uint ulPolynomial = 0x04c11db7;

		public Crc32()
		{
			for (uint i = 0; i < table.Length; i++)
			{
				table[i] = Reflect(i, 8) << 24;
				for (int j = 0; j < 8; j++)
					table[i] = (table[i] << 1) ^ ((table[i] & (1U << 31)) == 0 ? 0 : ulPolynomial);
				table[i] = Reflect(table[i], 32);
			}
		}
	}
}
