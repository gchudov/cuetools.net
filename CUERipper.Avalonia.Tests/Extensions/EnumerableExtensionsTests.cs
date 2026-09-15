using CUERipper.Avalonia.Extensions;

namespace CUERipper.Avalonia.Tests.Extensions
{
    public class EnumerableExtensionsTests
    {
        [Fact]
        public void None_WhenNull_ShouldReturnTrue()
        {
            // Arrange
            IEnumerable<string>? strings = null;

            // Act
            var result = strings.None();
            
            // Assert
            Assert.True(result);
        }

        [Fact]
        public void None_WhenEmpty_ShouldReturnTrue()
        {
            // Arrange
            IEnumerable<string> strings = [];

            // Act
            var result = strings.None();

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void NonePredicate_WhenNull_ShouldReturnTrue()
        {
            // Arrange
            IEnumerable<string>? strings = null;

            // Act
            var result = strings.None(x => x == "ABC");

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void NonePredicate_WhenEmpty_ShouldReturnTrue()
        {
            // Arrange
            IEnumerable<string>? strings = [];

            // Act
            var result = strings.None(x => x == "ABC");

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void PrependIf_WhenConditionMet_ShouldPrependItem()
        {
            // Arrange
            IEnumerable<string> strings = ["B", "C"];

            // Act
            var result = strings.PrependIf(true, "A");

            // Assert
            Assert.Equal(["A", "B", "C"], result);
        }

        [Fact]
        public void PrependIf_WhenConditionNotMet_ShouldReturnUnmodified()
        {
            // Arrange
            IEnumerable<string> strings = ["B", "C"];

            // Act
            var result = strings.PrependIf(false, "A");

            // Assert
            Assert.Equal(["B", "C"], result);
        }
    }
}
