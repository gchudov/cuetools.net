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
        private const int InterruptTimeout = 1000;

        private Task? _wrappedTask;
        private CancellationTokenSource _cts = new();

        public bool IsCompleted => _wrappedTask?.IsCompleted ?? true;
        public bool IsExecuting => !IsCompleted;

        public void Run(Func<CancellationToken, Task> function)
        {
            Interrupt();

            var token = _cts.Token;
            _wrappedTask = Task.Run(() => function(token), token)
            .ContinueWith((t) =>
            {
                if (t.IsFaulted)
                {
                    ExceptionDispatchInfo.Capture(t.Exception.InnerException
                        ?? t.Exception).Throw();
                };
            });
        }

        public void Interrupt()
        {
            if (_wrappedTask == null || _wrappedTask.IsCompleted) return;

            _cts.Cancel();

            try
            {
                _wrappedTask.Wait(InterruptTimeout);
            }
            catch (AggregateException ex)
            {
                if (!ex.InnerExceptions.Any(e => e is OperationCanceledException))
                    throw;
            }

            if (!_cts.TryReset()) _cts = new();
        }

        private bool WaitForCompletion()
        {
            if (_wrappedTask == null || _wrappedTask.IsCompleted) return true;

            try
            {
                return _wrappedTask.Wait(InterruptTimeout);
            }
            catch
            {
                return _wrappedTask.IsCompleted;
            }
        }

        private bool _disposed;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (!_cts.IsCancellationRequested) _cts.Cancel();

            if (WaitForCompletion())
            {
                _wrappedTask?.Dispose();
                _cts.Dispose();
            }

            GC.SuppressFinalize(this);
        }
    }
}
