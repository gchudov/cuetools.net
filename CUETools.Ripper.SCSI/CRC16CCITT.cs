using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Ripper.SCSI
{
	public enum InitialCrcValue { Zeros, NonZero1 = 0xffff, NonZero2 = 0x1D0F }

	class Crc16Ccitt
	{
		const ushort poly = 4129;
		ushort[] table = new ushort[256];
		ushort initialValue = 0;

		public ushort ComputeChecksum(byte[] bytes, int pos, int count)
		{
			ushort crc = this.initialValue;
			for (int i = pos; i < pos + count; i++)
			{
				crc = (ushort)((crc << 8) ^ table[((crc >> 8) ^ (0xff & bytes[i]))]);
			}
			return crc;
		}

		public byte[] ComputeChecksumBytes(byte[] bytes, int pos, int count)
		{
			ushort crc = ComputeChecksum(bytes, pos, count);
			return new byte[] { (byte)(crc >> 8), (byte)(crc & 0x00ff) };
		}

		public Crc16Ccitt(InitialCrcValue initialValue)
		{
			this.initialValue = (ushort)initialValue;
			ushort temp, a;
			for (int i = 0; i < table.Length; i++)
			{
				temp = 0;
				a = (ushort)(i << 8);
				for (int j = 0; j < 8; j++)
				{
					if (((temp ^ a) & 0x8000) != 0)
					{
						temp = (ushort)((temp << 1) ^ poly);
					}
					else
					{
						temp <<= 1;
					}
					a <<= 1;
				}
				table[i] = temp;
			}
		}
	}
}
