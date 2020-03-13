import termtables as tt

header = ["a", "bb", "ccc"]
data = [
    [1, 2, 3], [613.23236243236, 613.23236243236, 613.23236243236]
]

tt.print(
    data,
    header=header,
    style=tt.styles.ascii_thin,
    padding=(0, 1),
    alignment="lcr"
)