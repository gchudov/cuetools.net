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
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Controls.Templates;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Input.Platform;
using Avalonia.Interactivity;
using CUERipper.Avalonia.Views.UserControls.Abstractions;
#if NET47
using CUERipper.Avalonia.Compatibility;
#endif

namespace CUERipper.Avalonia.Views.UserControls;

/// <summary>
/// Provides extended capabilities for Avalonia grids
/// </summary>
/// <typeparam name="TColumnKey"></typeparam>
/// <typeparam name="TColumnDefinition"></typeparam>
public abstract class CUEGrid<TColumnKey, TColumnDefinition, TRowViewModel> : UserControl
    where TColumnKey : notnull, Enum
    where TColumnDefinition : GridColumnDefinition<TColumnKey>
    where TRowViewModel : notnull
{
    private DataGrid? _dataGrid;
    protected Dictionary<TColumnKey, TColumnDefinition> Columns = [];

    public CUEGrid() { }

    protected void InitGrid(DataGrid dataGrid)
    {
        _dataGrid = dataGrid;
        _dataGrid.AddHandler(KeyDownEvent, OnKeyDown, RoutingStrategies.Tunnel);

        foreach (var column in Columns)
        {
            _dataGrid.Columns.Add(column.Value.Create(column.Key, column.Value));
        }

        CreateContextMenu();
    }

    private void CreateContextMenu()
    {
        if (_dataGrid == null) return;

        var newMenuItem = (string text, Func<Task> clickHandler) => {
            var item = new MenuItem {
                Header = text
            };

            item.Click += (_,_) => clickHandler.Invoke();

            return item;
        };

        _dataGrid.ContextMenu ??= new ContextMenu();
        var menuItems = _dataGrid.ContextMenu.Items;

        menuItems.Add(newMenuItem("Copy column", OnCopyColumn));
        menuItems.Add(newMenuItem("Paste column", OnPasteColumn));
        menuItems.Add(newMenuItem("Copy range", OnCopyRange));
        menuItems.Add(newMenuItem("Paste range", OnPasteRange));
    }

    private IClipboard? Clipboard => TopLevel.GetTopLevel(this)?.Clipboard;

    private const string _columnSeparator = "\t";
    private string GetTableExportHeader()
    {
        if (_dataGrid == null) return string.Empty;

        var relevantColumns = Columns.Where(c => c.Value.Clipboard)
            .Select(c => c.Key);

        return string.Join(_columnSeparator, _dataGrid.Columns
            .Where(c => c.Tag is TColumnKey)
            .Where(c => relevantColumns.Contains((TColumnKey)c.Tag))
            .Select(c => c.Header switch
            {
                string str => str
                , TextBlock tb => tb.Text ?? string.Empty
                , _ => string.Empty
            }
        ));
    }

    private async Task OnCopyRange()
    {
        if (_dataGrid == null || Clipboard == null) return;

        var sb = new StringBuilder(GetTableExportHeader() + Environment.NewLine);
        foreach (var item in _dataGrid.SelectedItems)
        {
            if (item is not TRowViewModel row) continue;
            for (int i = 0; i < Columns.Count; ++i)
            {
                var column = Columns.ElementAt(i);
                if (!column.Value.Clipboard) continue;

                var property = column.Value.Binding;
                if (!string.IsNullOrWhiteSpace(property))
                {
                    var propInfo = typeof(TRowViewModel).GetProperty(property);
                    if (propInfo != null) sb.Append(propInfo.GetValue(row));
                }

                if (i != Columns.Count - 1) sb.Append(_columnSeparator);
            }

            sb.Append(Environment.NewLine);
        }

        await Clipboard.SetTextAsync(sb.ToString());
    }

    private async Task OnPasteRange()
    {
        if (_dataGrid == null || Clipboard == null) return;

        var text = await Clipboard.GetTextAsync();
        if (string.IsNullOrWhiteSpace(text)) return;

        var rows = text!.Split(Environment.NewLine)
            .Where(x => x != GetTableExportHeader());

        var clipboardColumns = Columns.Where(c => c.Value.Clipboard).ToList();
        var reflectedProperties = clipboardColumns.Where(c => !string.IsNullOrWhiteSpace(c.Value.Binding))
            .Select(c => typeof(TRowViewModel).GetProperty(c.Value.Binding!))
            .Where(c => c != null)
            .ToList();

        if (clipboardColumns.Count != reflectedProperties.Count) return;

        for (int rowIter = 0; rowIter < _dataGrid.SelectedItems.Count && rowIter < rows.Count(); ++rowIter)
        {
            var columns = rows.ElementAt(rowIter).Split(_columnSeparator);
            if (columns.Length < clipboardColumns.Count) continue;
            if (_dataGrid.SelectedItems[rowIter] is not TRowViewModel row) continue;

            for (int propIter = 0; propIter < reflectedProperties.Count; ++propIter)
            {
                if (!clipboardColumns.ElementAt(propIter).Value.ReadOnly)
                {
                    reflectedProperties[propIter]!.SetValue(row, columns[propIter]);
                }
            }
        }
    }

    private async Task OnPasteColumn()
    {
        if (_dataGrid == null || Clipboard == null) return;

        var text = await Clipboard.GetTextAsync();
        if (string.IsNullOrWhiteSpace(text)) return;

        var selectedColumn = _dataGrid.CurrentColumn;
        if (selectedColumn?.Tag is not TColumnKey key) return;

        if (!Columns.TryGetValue(key, out var column)) return;
        if (column.ReadOnly) return;

        var property = column.Binding;
        if (string.IsNullOrWhiteSpace(property)) return;

        var propInfo = typeof(TRowViewModel).GetProperty(property);
        if (propInfo == null) return;

        foreach (TRowViewModel row in _dataGrid.SelectedItems)
        {
            propInfo.SetValue(row, text);
        }
    }

    private async Task OnCopyColumn()
    {
        if (_dataGrid == null || Clipboard == null) return;

        var selectedColumn = _dataGrid.CurrentColumn;
        if (selectedColumn?.Tag is not TColumnKey key) return;

        // Pick the last selected item
        var selected = _dataGrid.SelectedItem;
        if (selected is not TRowViewModel row) return;

        if (!Columns.TryGetValue(key, out var column)) return;
        if (!column.Clipboard) return;

        var property = column.Binding;
        if (string.IsNullOrWhiteSpace(property)) return;

        var propInfo = typeof(TRowViewModel).GetProperty(property);
        if (propInfo == null) return;

        var text = propInfo.GetValue(row) as string;
        if (!string.IsNullOrWhiteSpace(text))
        {
            await Clipboard.SetTextAsync(text);
        }
    }

    private void OnDeleteColumn()
    {
        if (_dataGrid == null || Clipboard == null) return;

        var selectedColumn = _dataGrid.CurrentColumn;
        if (selectedColumn?.Tag is not TColumnKey key) return;

        if (!Columns.TryGetValue(key, out var column)) return;
        if (column.ReadOnly) return;

        var property = column.Binding;
        if (string.IsNullOrWhiteSpace(property)) return;

        var propInfo = typeof(TRowViewModel).GetProperty(property);
        if (propInfo == null) return;

        foreach (TRowViewModel row in _dataGrid.SelectedItems)
        {
            propInfo.SetValue(row, string.Empty);
        }
    }

    private static readonly HashSet<Key> _ignoredKeys =
    [
        Key.LeftCtrl
        , Key.RightCtrl
        , Key.LeftAlt
        , Key.RightAlt
        , Key.LeftShift
        , Key.RightShift
        , Key.Tab
        , Key.Enter
        , Key.Escape
        , Key.LWin
        , Key.RWin
        , Key.CapsLock
        , Key.NumLock
        , Key.Scroll
        , Key.Pause
        , Key.Home
        , Key.End
        , Key.Insert
        , Key.PageUp
        , Key.PageDown
        , Key.PrintScreen
        , Key.Up
        , Key.Down
        , Key.Left
        , Key.Right
        , Key.F1
        , Key.F2
        , Key.F3
        , Key.F4
        , Key.F5
        , Key.F6
        , Key.F7
        , Key.F8
        , Key.F9
        , Key.F10
        , Key.F11
        , Key.F12
    ];

    private async void OnKeyDown(object? sender, KeyEventArgs e)
    {
        if (sender is not DataGrid grid || grid.SelectedItem == null) return;
        if (_ignoredKeys.Contains(e.Key)) return;

        if (e.KeyModifiers.HasFlag(KeyModifiers.Alt) ||
            e.KeyModifiers.HasFlag(KeyModifiers.Meta))
            return;

        if (e.KeyModifiers.HasFlag(KeyModifiers.Control))
        {
            await OnCtrlKeyDown(grid, e);
            return;
        }

        if (e.Key == Key.Delete && !grid.IsReadOnly)
        {
            e.Handled = true;

            OnDeleteColumn();
            return;
        }

        if (!grid.IsReadOnly) grid.BeginEdit();
    }

    private async Task OnCtrlKeyDown(DataGrid grid, KeyEventArgs e)
    {
        switch (e.Key)
        {
            case Key.C:
                e.Handled = true;
                await (grid.CurrentColumn != null ? OnCopyColumn() : OnCopyRange());
                break;
            case Key.V when !grid.IsReadOnly:
                e.Handled = true;
                await (grid.CurrentColumn != null ? OnPasteColumn() : OnPasteRange());
                break;
            case Key.X when !grid.IsReadOnly:
                e.Handled = true;
                await OnCopyColumn();
                OnDeleteColumn();
                break;
            case Key.A:
                grid.SelectedIndex = -1;
                grid.Focus();
                break;
        }
    }

    private static object GetHeaderFromDefinition(TColumnDefinition definition)
    {
        if (!definition.HeaderBinding) return definition.Header;

        var textBlock = new TextBlock();
        textBlock.Bind(TextBlock.TextProperty, new Binding(definition.Header, BindingMode.OneTime));

        return textBlock;
    }

    protected static DataGridColumn CreateTextColumn(TColumnKey key, TColumnDefinition definition)
        => new DataGridTextColumn {
            Tag = key
            , Header = GetHeaderFromDefinition(definition)
            , Binding = definition.Binding != null ? new Binding(definition.Binding,
                definition.ReadOnly ? BindingMode.OneWay : BindingMode.TwoWay) : null
            , IsReadOnly = definition.ReadOnly
        };

    protected static DataGridColumn CreateProgressColumn(TColumnKey key, TColumnDefinition definition)
    {
        var progressBarTemplate = new FuncDataTemplate<object>((_, _) => {
            var progressBar = new ProgressBar {
                Minimum = 0
                , Maximum = 100
                , Height = 18
                , Width = 80
                , ShowProgressText = true
            };

            if (definition.Binding != null)
            {
                progressBar.Bind(ProgressBar.ValueProperty, new Binding(definition.Binding, BindingMode.OneWay));
            }

            return progressBar;
        }, true);

        return new DataGridTemplateColumn {
            Tag = key
            , Header = GetHeaderFromDefinition(definition)
            , CellTemplate = progressBarTemplate
        };
    }
}