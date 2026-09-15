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

namespace CUERipper.Avalonia.ViewModels.Bindings.OptionProxies
{
    public static class OptionProxyFactory
    {
        public static IOptionProxy Create(Type type
            , string name
            , object defaultValue
            , object accessor)
        {
            if (type == typeof(int))
            {
                if (defaultValue.GetType() != typeof(int) || accessor.GetType() != typeof(Accessor<int>))
                    throw new InvalidOperationException("Can't create int proxy.");

                return new IntOptionProxy(name, (int)defaultValue, (Accessor<int>)accessor);
            }
            else if (type == typeof(string))
            {
                if (defaultValue.GetType() != typeof(string) || accessor.GetType() != typeof(Accessor<string>))
                    throw new InvalidOperationException("Can't create string proxy.");

                return new StringOptionProxy(name, (string)defaultValue, (Accessor<string>)accessor);
            }
            else if (type == typeof(bool))
            {
                if (defaultValue.GetType() != typeof(bool) || accessor.GetType() != typeof(Accessor<bool>))
                    throw new InvalidOperationException("Can't create bool proxy.");

                return new BoolOptionProxy(name, (bool)defaultValue, (Accessor<bool>)accessor);
            }
            else if (type.IsEnum) return CreateEnum(type, name, defaultValue, accessor);

            throw new NotSupportedException($"No proxy available for type '{type}'.");
        }

        private static IOptionProxy CreateEnum(Type type
            , string name
            , object defaultValue
            , object accessor)
        {
            Type proxyType = typeof(EnumOptionProxy<>);
            Type specificProxyType = proxyType.MakeGenericType(type);

            return Activator.CreateInstance(specificProxyType
                , name
                , defaultValue
                , accessor
            ) as IOptionProxy ?? throw new NullReferenceException($"Failed to create enum proxy {type}.");
        }
    }
}
