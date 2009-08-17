using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs
{
	public class Crc16
	{
		ushort[] table = new ushort[256];

		public ushort ComputeChecksum(byte[] bytes, int pos, int count)
		{
			ushort crc = 0;
			for (int i = pos; i < pos + count; i++)
			{
				crc = (ushort)(((crc << 8) & 0xffff) ^ table[(crc >> 8) ^ bytes[i]]);
			}
			return crc;
		}

		public unsafe ushort ComputeChecksum(byte* bytes, int pos, int count)
		{
			ushort crc = 0;
			fixed (ushort* t = table)
				for (int i = pos; i < pos + count; i++)
					crc = (ushort)(((crc << 8) & 0xffff) ^ t[(crc >> 8) ^ bytes[i]]);
			return crc;
		}

		public Crc16()
		{
			int bits = 16;
			int poly16 = 0x8005;
			int poly = (poly16 + (1 << bits));
			for (ushort i = 0; i < table.Length; i++)
			{
				int crc = i;
				for (int j = 0; j < bits; j++)
				{
					if ((crc & (1U << (bits - 1))) != 0)
						crc = ((crc << 1) ^ poly);
					else
						crc <<= 1;
				}
			        //table[i] = (crc & ((1<<bits)-1));
				table[i] = (ushort)(crc & 0xffff);
			}
		}
	}
}
