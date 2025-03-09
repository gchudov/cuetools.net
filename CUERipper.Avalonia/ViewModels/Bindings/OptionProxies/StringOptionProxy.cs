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

namespace CUERipper.Avalonia.ViewModels.Bindings.OptionProxies
{
    public class StringOptionProxy : OptionProxy<string>
    {
        public StringOptionProxy(string name, string defaultValue, Accessor<string> accessor)
            : base(name, defaultValue, accessor)
        {
            Value = string.IsNullOrWhiteSpace(Value) ? defaultValue : Value;
        }

        protected override string GetStringFromValue(string val)
            => val;

        protected override string GetValueFromString(string str)
            => str;
    }
}
