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
using System.Threading.Tasks;
using System.Threading;
using System.Linq;
using CUERipper.Avalonia.Compatibility;
using System.Runtime.ExceptionServices;

namespace CUERipper.Avalonia.Utilities
{
    /// <summary>
    /// A wrapper around a 'Task' that ensures only one task runs at a time.  
    /// If a new task is assigned, the current one is canceled before starting the new one.  
    /// If the current task can't be canceled (likely due to a deadlock), it will be ignored, and the next task will start.
    /// </summary>
    public sealed class InterruptibleJob : IDisposable
    {
        private Task? _wrappedTask;
        private CancellationTokenSource _cts = new();

        public bool IsCompleted => _wrappedTask?.IsCompleted ?? true;
        public bool IsExecuting => !IsCompleted;

        public void Run(Func<CancellationToken, Task> function)
        {
            TryFinish();

            _wrappedTask = Task.Factory.StartNew(async () =>
            {
                await function(_cts.Token);
            }, _cts.Token, TaskCreationOptions.LongRunning, TaskScheduler.Default)
            .ContinueWith((t) =>
            {
                if (t.IsFaulted)
                {
                    ExceptionDispatchInfo.Capture(t.Exception.InnerException
                        ?? t.Exception).Throw();
                };
            });
        }

        private void TryFinish()
        {
            if (_wrappedTask == null || _wrappedTask.IsCompleted) return;

            _cts.Cancel();

            try
            {
                _wrappedTask.Wait(1000);
            }
            catch (AggregateException ex)
            {
                if (!ex.InnerExceptions.Select(e => e.GetType()).Contains(typeof(TaskCanceledException)))
                    throw;
            }

            if (!_cts.TryReset())
            {
                _cts.Dispose();
                _cts = new();
            }
        }

        private bool _disposed;
        public void Dispose()
        {
            if (_disposed == true) return;
            _disposed = true;

            if (!_cts.IsCancellationRequested) _cts.Cancel();

            if (_wrappedTask != null)
            {
                if (!_wrappedTask.IsCompleted)
                {
                    try
                    {
                        _wrappedTask.Wait(1000);
                    }
                    catch
                    {
                        // ..
                    }
                }

                _wrappedTask.Dispose();
            }

            _cts.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}
