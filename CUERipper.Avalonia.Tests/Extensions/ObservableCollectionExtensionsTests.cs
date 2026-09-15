using CUERipper.Avalonia.Extensions;
using System.Collections.ObjectModel;

namespace CUERipper.Avalonia.Tests.Extensions
{
    public class ObservableCollectionExtensionsTests
    {
        [Fact]
        public void MoveAll_WhenSourceIsNull_ShouldNotModifyDestination()
        {
            // Arrange
            ObservableCollection<string>? colSrc = null;
            ObservableCollection<string> colDest = ["One", "Two", "Three"];

            // Act
            colSrc.MoveAll(colDest);

            // Assert
            Assert.Null(colSrc);
            Assert.Equal(["One", "Two", "Three"], colDest);
        }

        [Theory]
        [InlineData(new string[] { "One", "Two" }, new string[] { "Three" }, new string[] { "Three", "One", "Two" })]
        [InlineData(new string[] { "One" }, new string[] { }, new string[] { "One" })]
        [InlineData(new string[] { }, new string[] { "One", "Two", "Three" }, new string[] { "One", "Two", "Three" })]
        public void MoveAll_ShouldMoveAllItemsFromSourceToDestination(string[] src, string[] dest, string[] expectedDest)
        {
            // Arrange
            ObservableCollection<string> colSrc = [..src];
            ObservableCollection<string> colDest = [..dest];

            // Act
            colSrc.MoveAll(colDest);

            // Assert
            Assert.Empty(colSrc);
            Assert.Equal(expectedDest, colDest);
        }
    }
}
