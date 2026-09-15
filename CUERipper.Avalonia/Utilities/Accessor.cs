/* 
    Taken from: https://stackoverflow.com/a/43498938
    License: https://creativecommons.org/licenses/by-sa/3.0/
    Original author: https://stackoverflow.com/users/442204/sven
    Modified by: Max Visser

    Modified for CUERipper, under the same license.
*/

using System.Linq.Expressions;
using System.Reflection;
using System;

namespace CUERipper.Avalonia.Utilities
{
    public class Accessor<T>
    {
        private readonly Func<T> _getter;
        private readonly Action<T>? _setter;

        public bool IsReadOnly => _setter == null;

        /// <exception cref="ArgumentException">Failed to retrieve getter or not a property or field.</exception>
        public Accessor(Expression<Func<T>> expr)
        {
            var memberExpression = (MemberExpression)expr.Body;
            var instanceExpression = memberExpression.Expression;
            var parameter = Expression.Parameter(typeof(T));

            if (memberExpression.Member is PropertyInfo propertyInfo)
            {
                var getMethod = propertyInfo.GetGetMethod()
                    ?? throw new ArgumentException("No getter found, a getter is required.");

                _getter = Expression.Lambda<Func<T>>(Expression.Call(instanceExpression, getMethod)).Compile();

                var setMethod = propertyInfo.GetSetMethod();
                if (setMethod != null)
                {
                    _setter = Expression.Lambda<Action<T>>(Expression.Call(instanceExpression, setMethod, parameter), parameter).Compile();
                }
            }
            else if (memberExpression.Member is FieldInfo fieldInfo)
            {
                _getter = Expression.Lambda<Func<T>>(Expression.Field(instanceExpression, fieldInfo)).Compile();
                _setter = Expression.Lambda<Action<T>>(Expression.Assign(memberExpression, parameter), parameter).Compile();
            }
            else
            {
                throw new ArgumentException("Not a field or property.");
            }
        }

        /// <exception cref="ArgumentNullException">Argument not provided.</exception>
        public Accessor(Expression instanceExpression, MethodInfo? getMethod, MethodInfo? setMethod)
        {
            if (getMethod == null) throw new ArgumentNullException(nameof(getMethod));

            var parameter = Expression.Parameter(typeof(T));
            _getter = Expression.Lambda<Func<T>>(Expression.Call(instanceExpression, getMethod)).Compile();

            if (setMethod != null)
            {
                _setter = Expression.Lambda<Action<T>>(Expression.Call(instanceExpression, setMethod, parameter), parameter).Compile();
            }
        }

        // Warning: Calls the code above!
        public static object? CreateAccessor(Type type, Expression instanceExpression, MethodInfo? getMethod, MethodInfo? setMethod)
        {
            Type accessor = typeof(Accessor<>);
            Type accessorWithType = accessor.MakeGenericType(type);

            return Activator.CreateInstance(accessorWithType
                , instanceExpression
                , getMethod
                , setMethod
            );
        }

        public T Get() => _getter != null ? _getter() : throw new InvalidOperationException("Getter not found.");
        public void Set(T value) => _setter?.Invoke(value);
    }
}
