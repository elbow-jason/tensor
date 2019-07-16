defmodule MatrixTest do
  use ExUnit.Case
  use Tensor
  doctest Matrix

  def sequential(rows, columns) do
    Stream.iterate(1, fn v -> v + 1 end)
    |> Enum.take(rows * columns)
    |> Enum.chunk_every(columns)
    |> Matrix.new(rows, columns)
  end

  test "Inspect" do
    matrix = Matrix.new([[1, 2], [3, 4]], 2, 2)

    assert Inspect.inspect(matrix, []) == """
           #Matrix<(2×2)
           ┌                 ┐
           │       1,       2│
           │       3,       4│
           └                 ┘
           >
           """
  end

  test "identity_matrix" do
    inspect(
      Matrix.identity_matrix(3) == """
      #Matrix<(3×3)
      ┌                          ┐
      │       1,       0,       0│
      │       0,       1,       0│
      │       0,       0,       1│
      └                          ┘
      >
      """
    )
  end

  test "transpose |> transpose is the same as original" do
    matrix = Matrix.new([[1, 2], [3, 4]], 2, 2)
    assert matrix |> Matrix.transpose() |> Matrix.transpose() == matrix
  end

  test "Scalar Addition" do
    matrix = Matrix.new([[1, 2], [3, 4]], 2, 2)
    result = Matrix.new([[3, 4], [5, 6]], 2, 2, 2)

    assert Matrix.add(matrix, 2) == result
  end

  test "scalar addition is commutative with transposition" do
    matrix = Matrix.new([[1, 2], [3, 4]], 2, 2)

    assert matrix |> Matrix.transpose() |> Matrix.add(2) ==
             matrix |> Matrix.add(2) |> Matrix.transpose()
  end

  test "chess" do
    board_as_list = [
      ["♜", "♞", "♝", "♛", "♚", "♝", "♞", "♜"],
      ["♟", "♟", "♟", "♟", "♟", "♟", "♟", "♟"],
      [" ", " ", " ", " ", " ", " ", " ", " "],
      [" ", " ", " ", " ", " ", " ", " ", " "],
      [" ", " ", " ", " ", " ", " ", " ", " "],
      [" ", " ", " ", " ", " ", " ", " ", " "],
      ["♙", "♙", "♙", "♙", "♙", "♙", "♙", "♙"],
      ["♖", "♘", "♗", "♕", "♔", "♗", "♘", "♖"]
    ]

    matrix = Matrix.new(board_as_list, 8, 8)

    assert inspect(matrix) ==
             """
             #Matrix<(8×8)
             ┌                                                                       ┐
             │     "♜",     "♞",     "♝",     "♛",     "♚",     "♝",     "♞",     "♜"│
             │     "♟",     "♟",     "♟",     "♟",     "♟",     "♟",     "♟",     "♟"│
             │     " ",     " ",     " ",     " ",     " ",     " ",     " ",     " "│
             │     " ",     " ",     " ",     " ",     " ",     " ",     " ",     " "│
             │     " ",     " ",     " ",     " ",     " ",     " ",     " ",     " "│
             │     " ",     " ",     " ",     " ",     " ",     " ",     " ",     " "│
             │     "♙",     "♙",     "♙",     "♙",     "♙",     "♙",     "♙",     "♙"│
             │     "♖",     "♘",     "♗",     "♕",     "♔",     "♗",     "♘",     "♖"│
             └                                                                       ┘
             >
             """
  end

  describe "trace/1" do
    test "works for square matrices" do
      matrix = sequential(2, 2)
      assert Matrix.trace(matrix) == 5
    end

    test "raises for non-square matrices" do
      matrix = sequential(2, 3)
      err = assert_raise(Tensor.ArithmeticError, fn -> Matrix.trace(matrix) end)

      expected_message = """
      Matrix.trace/1 is not defined for non-square matrices!

      height: 2
      width: 3
      """

      assert err == %Tensor.ArithmeticError{message: expected_message}
    end
  end

  describe "power/2" do
    test "works for a square matrix" do
      matrix = sequential(2, 2)

      assert matrix |> Matrix.power(2) |> Matrix.to_list() == [
               [7, 10],
               [15, 22]
             ]
    end

    test "raises for non-square matrices" do
      matrix = sequential(3, 2)

      err = assert_raise(Tensor.ArithmeticError, fn -> Matrix.power(matrix, 2) end)

      expected_message = """
      Cannot compute Matrix.power with non-square matrices!

      height: 3
      width: 2
      exponent: 2
      """

      assert err == %Tensor.ArithmeticError{message: expected_message}
    end
  end

  describe "product/2" do
    test "matrix productiplication with the identity matrix results in same matrix" do
      m1 = Matrix.new([[2, 3, 4], [1, 0, 0]], 2, 3)
      mid = Matrix.identity_matrix(3)

      assert Matrix.product(m1, mid) == m1
    end

    test "works for left and right matrices when left width is the same as right height" do
      m1 = Matrix.new([[2, 3, 4], [1, 0, 0]], 2, 3)
      m2 = Matrix.new([[0, 1000], [1, 100], [0, 10]], 3, 2)

      assert Matrix.product(m1, m2) |> inspect == """
             #Matrix<(2×2)
             ┌                 ┐
             │       3,    2340│
             │       0,    1000│
             └                 ┘
             >
             """
    end

    test "raises for matrices with mismatched dimensions" do
      m1 = Matrix.new([[1, 2, 3, 4]], 1, 4)
      m2 = Matrix.new([[0, 1000], [1, 100], [0, 10]], 3, 2)

      err = assert_raise(Tensor.ArithmeticError, fn -> Matrix.product(m1, m2) end)

      expected_message = """
      Cannot compute Matrix.product if the width of matrix `a` does not match the height of matrix `b`!

      height_a: 1
      width_a: 4
      height_b: 3
      width_b: 2
      """

      assert err == %Tensor.ArithmeticError{message: expected_message}
    end
  end
end
