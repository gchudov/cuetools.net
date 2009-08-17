using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs
{

	public class Crc8
	{
		const ushort poly8 = 0x07;
		ushort[] table = new ushort[256];

		public byte ComputeChecksum(byte[] bytes, int pos, int count)
		{
			ushort crc = 0;
			for (int i = pos; i < pos + count; i++)
				crc = table[crc ^ bytes[i]];
			return (byte) crc;
		}

		public unsafe byte ComputeChecksum(byte * bytes, int pos, int count)
		{
			ushort crc = 0;
			for (int i = pos; i < pos + count; i++)
				crc = table[crc ^ bytes[i]];
			return (byte)crc;
		}

		public Crc8()
		{
			int bits = 8;
			ushort poly = (ushort) (poly8 + (1U << bits));
			for (ushort i = 0; i < table.Length; i++)
			{
				ushort crc = i;
				for (int j = 0; j < bits; j++)
				{
					if ((crc & (1U << (bits - 1))) != 0)
						crc = (ushort)((crc << 1) ^ poly);
					else
						crc <<= 1;
				}
				table[i] = (ushort)(crc & 0x00ff);
			}
		}
	}
}
