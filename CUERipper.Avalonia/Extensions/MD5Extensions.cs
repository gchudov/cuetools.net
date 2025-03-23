#region Copyright (C) 2025 Max Visser
/*
    Copyright (C) 2025 Max Visser

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, see <https://www.gnu.org/licenses/>.
*/
#endregion
using System.Security.Cryptography;
using System.Text;

namespace CUERipper.Avalonia.Extensions
{
    public static class MD5Extensions
    {
        /// <summary>
        /// Generates an MD5 hash.
        /// The resulting hash is 16 bytes long and relatively unique.
        /// </summary>
        /// <param name="input">The bytes to hash.</param>
        /// <returns>A 16-byte MD5 hash as a hexidecimal string.</returns>
        public static string ComputeHashAsString(this MD5 md5, byte[] input, string format = "X2")
        {
            var hash = md5.ComputeHash(input);

            var stringBuilder = new StringBuilder();
            for (int i = 0; i < hash.Length; ++i)
            {
                stringBuilder.Append(hash[i].ToString(format));
            }

            return stringBuilder.ToString();
        }

        /// <summary>
        /// Generates an MD5 hash.
        /// The resulting hash is 16 bytes long and relatively unique.
        /// </summary>
        /// <param name="input">The bytes to hash.</param>
        /// <returns>A 16-byte MD5 hash as a hexidecimal string.</returns>
        public static string ComputeHashAsString(this MD5 md5, string input, string format = "X2")
            => ComputeHashAsString(md5, Encoding.UTF8.GetBytes(input), format);
    }
}
