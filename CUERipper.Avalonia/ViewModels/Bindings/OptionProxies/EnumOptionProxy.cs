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
using CUERipper.Avalonia.Utilities;
using CUERipper.Avalonia.ViewModels.Bindings.OptionProxies.Abstractions;
using System;
using System.ComponentModel;

namespace CUERipper.Avalonia.ViewModels.Bindings.OptionProxies
{
    public class EnumOptionProxy<T> : OptionProxy<T> where T : Enum
    {
        public EnumOptionProxy(string name, T defaultValue, Accessor<T> accessor)
            : base(name, defaultValue, accessor)
        {
            Options = [.. Enum.GetNames(typeof(T))];
        }

        protected override string GetStringFromValue(T val)
            => Enum.GetName(typeof(T), val)
                    ?? throw new InvalidEnumArgumentException("Can't get name of enum value.");

        protected override T GetValueFromString(string str)
            => (T)Enum.Parse(typeof(T), str);
    }
}