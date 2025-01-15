#!/usr/bin/env python
# coding: utf-8


# CellIDから行番号及び列番号を算出する関数（行, 列）
# from get_cellid_within_range import get_row_col
# row, col = get_row_col(5161)
def get_row_col(cell_id):
    row = (cell_id - 1) // 100 + 1
    col = (cell_id - 1) % 100 + 1
    return row, col

# 指定された中心セルIDから、範囲内のセルIDを取得する関数。
# from get_cellid_within_range import get_cells_in_range
# cells_in_range = get_cells_in_range(5161, 20)
def get_cells_in_range(center_cell_id, range_size):
        
    center_row, center_col = get_row_col(center_cell_id)
    mid = range_size // 2  # 中心からの距離

    cells_in_range = []
    for row in range(center_row - mid, center_row + mid + 1):
        for col in range(center_col - mid, center_col + mid + 1):
            try:
                if 1 <= row <= 100 and 1 <= col <= 100:
                    # セルIDを計算
                    cell_id = (row - 1) * 100 + col
                    cells_in_range.append(cell_id)
                else:
                    raise ValueError(f"Cell out of bounds: row={row}, col={col}")
            except ValueError as e:
                print(f"Error: {e}")

    return cells_in_range

# セルID範囲内の行番号と列番号を取得する関数
# from get_cellid_within_range import get_row_col_for_cells_in_range
# rows, cols = get_row_col_for_cells_in_range(5161, 20)
def get_row_col_for_cells_in_range(center_cell_id, range_size):
    # 範囲内のセルIDを取得
    cells_in_range = get_cells_in_range(center_cell_id, range_size)
    
    # 行番号と列番号をそれぞれのリストに格納
    rows = []
    cols = []
    
    for cell_id in cells_in_range:
        row, col = get_row_col(cell_id)
        rows.append(row - 1)
        cols.append(col)

    # 重複を削除（順序を保つために sorted() でソート）
    rows = sorted(set(rows))
    cols = sorted(set(cols))
    
    return rows, cols

