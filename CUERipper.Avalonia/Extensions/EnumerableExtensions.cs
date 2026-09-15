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
using CUERipper.Avalonia.Compatibility;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CUERipper.Avalonia.Extensions
{
    public static class EnumerableExtensions
    {
        /// <summary>
        /// Silly replacement for !Any for readability
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <returns>inverted .Any() with an additional null check</returns>
        public static bool None<T>(this IEnumerable<T>? src)
            => src == null || !src.Any();

        public static bool None<T>(this IEnumerable<T>? src, Func<T, bool> predicate)
            => src == null || !src.Any(predicate);

        public static IEnumerable<T> PrependIf<T>(this IEnumerable<T> src, bool condition, T item)
        {
            if (condition)
            {
                src = src.Prepend(item);
            }

            return src;
        }
    }
}
