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
using Avalonia.Controls;
using CUERipper.Avalonia.Exceptions;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Utilities;
using CUERipper.Avalonia.ViewModels;
using CUERipper.Avalonia.ViewModels.Bindings.OptionProxies;
using CUETools.Codecs;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Threading.Tasks;

namespace CUERipper.Avalonia;

public partial class EncoderOptionsDialog : Window
{
    public EncoderOptionsDialogViewModel ViewModel => DataContext as EncoderOptionsDialogViewModel
        ?? throw new ViewModelMismatchException(typeof(EncoderOptionsDialogViewModel), DataContext?.GetType());

    public required IAudioEncoderSettings EncoderSettings { get; init; }
    public EncoderOptionsDialog()
    {
        InitializeComponent();
        DataContextChanged += OnDataContextChanged;
    }

    private void OnDataContextChanged(object? sender, EventArgs e)
    {
        ViewModel.Options.Clear();

        var properties = GetBrowsableProperties(EncoderSettings);
        foreach(var property in properties)
        {
            var accessorInstance = Accessor<object?>.CreateAccessor(property.Type
                , Expression.Constant(EncoderSettings)
                , property.GetMethod
                , property.SetMethod
            );

            if (accessorInstance == null) continue;

            var defaultValue = (property.DefaultValue ?? (property.Type.IsValueType
                ? Activator.CreateInstance(property.Type)
                : (property.Type == typeof(string) ? string.Empty : null)
            )) ?? throw new ArgumentNullException(nameof(property.DefaultValue), "Null can't be a default value for OptionProxy");

            var proxyInstance = OptionProxyFactory.Create(property.Type
                , property.DisplayName
                , defaultValue
                , accessorInstance
            );

            if (proxyInstance != null)
            {
                ViewModel.Options.Add(proxyInstance);
            }
        }
    }

    public static List<PropertyMetadata> GetBrowsableProperties(object obj)
    {
        return obj.GetType()
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.GetCustomAttributes<BrowsableAttribute>().None(attr => !attr.Browsable))
            .Select(p => new PropertyMetadata
            {     
                Name = p.Name
                , DisplayName = p.GetCustomAttribute<DisplayNameAttribute>()?.DisplayName ?? p.Name
                , Description = p.GetCustomAttribute<DescriptionAttribute>()?.Description
                , DefaultValue = p.GetCustomAttribute<DefaultValueAttribute>()?.Value
                , Type = p.PropertyType
                , GetMethod = p.GetGetMethod()
                , SetMethod = p.GetSetMethod()
            })
            .ToList();
    }

    public class PropertyMetadata
    {
        public required string Name { get; init; }
        public required string DisplayName { get; init; }
        public string? Description { get; set; }
        public object? DefaultValue { get; set; }
        public required Type Type { get; init; }
        public MethodInfo? GetMethod { get; set; }
        public MethodInfo? SetMethod { get; set; }
    }

    public static async Task CreateAsync(Window owner, IAudioEncoderSettings encoderSettings)
    {
        var encodingSettingsWindow = new EncoderOptionsDialog()
        {
            Owner = owner
            , EncoderSettings = encoderSettings
            , DataContext = new EncoderOptionsDialogViewModel()
        };

        await encodingSettingsWindow.ShowDialog(owner, lockParent: true);
    }
}