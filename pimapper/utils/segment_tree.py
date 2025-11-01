class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy_add = [0] * (4 * self.n)
        self.lazy_set = [None] * (4 * self.n)

        def build(node, start, end):
            if start == end:
                self.tree[node] = arr[start]
            else:
                mid = (start + end) // 2
                build(2 * node + 1, start, mid)
                build(2 * node + 2, mid + 1, end)
                self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

        build(0, 0, self.n - 1)

    def query(self, l, r):
        def helper(node, start, end):
            if self.lazy_set[node] is not None:
                self.tree[node] = self.lazy_set[node] * (end - start + 1)
                if start != end:
                    self.lazy_set[2 * node + 1] = self.lazy_set[node]
                    self.lazy_set[2 * node + 2] = self.lazy_set[node]
                    self.lazy_add[2 * node + 1] = 0
                    self.lazy_add[2 * node + 2] = 0
                self.lazy_set[node] = None

            if self.lazy_add[node] != 0:
                self.tree[node] += self.lazy_add[node] * (end - start + 1)
                if start != end:
                    self.lazy_add[2 * node + 1] += self.lazy_add[node]
                    self.lazy_add[2 * node + 2] += self.lazy_add[node]
                self.lazy_add[node] = 0

            if start > r or end < l:
                return 0
            if start >= l and end <= r:
                return self.tree[node]
            mid = (start + end) // 2
            return helper(2 * node + 1, start, mid) + helper(2 * node + 2, mid + 1, end)

        return helper(0, 0, self.n - 1)

    def range_add(self, l, r, val):
        def helper(node, start, end):
            if self.lazy_set[node] is not None:
                self.tree[node] = self.lazy_set[node] * (end - start + 1)
                if start != end:
                    self.lazy_set[2 * node + 1] = self.lazy_set[node]
                    self.lazy_set[2 * node + 2] = self.lazy_set[node]
                    self.lazy_add[2 * node + 1] = 0
                    self.lazy_add[2 * node + 2] = 0
                self.lazy_set[node] = None

            if self.lazy_add[node] != 0:
                self.tree[node] += self.lazy_add[node] * (end - start + 1)
                if start != end:
                    self.lazy_add[2 * node + 1] += self.lazy_add[node]
                    self.lazy_add[2 * node + 2] += self.lazy_add[node]
                self.lazy_add[node] = 0

            if start > r or end < l:
                return
            if start >= l and end <= r:
                self.lazy_add[node] += val
                self.tree[node] += val * (end - start + 1)
                if start != end:
                    self.lazy_add[2 * node + 1] += val
                    self.lazy_add[2 * node + 2] += val
                self.lazy_add[node] = 0
                return
            mid = (start + end) // 2
            helper(2 * node + 1, start, mid)
            helper(2 * node + 2, mid + 1, end)

            def push(n, s, e):
                if self.lazy_set[n] is not None:
                    self.tree[n] = self.lazy_set[n] * (e - s + 1)
                    if s != e:
                        self.lazy_set[2 * n + 1] = self.lazy_set[n]
                        self.lazy_set[2 * n + 2] = self.lazy_set[n]
                        self.lazy_add[2 * n + 1] = 0
                        self.lazy_add[2 * n + 2] = 0
                    self.lazy_set[n] = None
                if self.lazy_add[n] != 0:
                    self.tree[n] += self.lazy_add[n] * (e - s + 1)
                    if s != e:
                        self.lazy_add[2 * n + 1] += self.lazy_add[n]
                        self.lazy_add[2 * n + 2] += self.lazy_add[n]
                    self.lazy_add[n] = 0

            push(2 * node + 1, start, mid)
            push(2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

        helper(0, 0, self.n - 1)

    def range_set(self, l, r, val):
        def helper(node, start, end):
            if self.lazy_set[node] is not None:
                self.tree[node] = self.lazy_set[node] * (end - start + 1)
                if start != end:
                    self.lazy_set[2 * node + 1] = self.lazy_set[node]
                    self.lazy_set[2 * node + 2] = self.lazy_set[node]
                    self.lazy_add[2 * node + 1] = 0
                    self.lazy_add[2 * node + 2] = 0
                self.lazy_set[node] = None

            if self.lazy_add[node] != 0:
                self.tree[node] += self.lazy_add[node] * (end - start + 1)
                if start != end:
                    self.lazy_add[2 * node + 1] += self.lazy_add[node]
                    self.lazy_add[2 * node + 2] += self.lazy_add[node]
                self.lazy_add[node] = 0

            if start > r or end < l:
                return
            if start >= l and end <= r:
                self.lazy_set[node] = val
                self.lazy_add[node] = 0
                self.tree[node] = val * (end - start + 1)
                if start != end:
                    self.lazy_set[2 * node + 1] = val
                    self.lazy_set[2 * node + 2] = val
                    self.lazy_add[2 * node + 1] = 0
                    self.lazy_add[2 * node + 2] = 0
                self.lazy_set[node] = None
                return
            mid = (start + end) // 2
            helper(2 * node + 1, start, mid)
            helper(2 * node + 2, mid + 1, end)

            def push(n, s, e):
                if self.lazy_set[n] is not None:
                    self.tree[n] = self.lazy_set[n] * (e - s + 1)
                    if s != e:
                        self.lazy_set[2 * n + 1] = self.lazy_set[n]
                        self.lazy_set[2 * n + 2] = self.lazy_set[n]
                        self.lazy_add[2 * n + 1] = 0
                        self.lazy_add[2 * n + 2] = 0
                    self.lazy_set[n] = None
                if self.lazy_add[n] != 0:
                    self.tree[n] += self.lazy_add[n] * (e - s + 1)
                    if s != e:
                        self.lazy_add[2 * n + 1] += self.lazy_add[n]
                        self.lazy_add[2 * n + 2] += self.lazy_add[n]
                    self.lazy_add[n] = 0

            push(2 * node + 1, start, mid)
            push(2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

        helper(0, 0, self.n - 1)

    def point_add(self, idx, val):
        self.range_add(idx, idx, val)

    def point_set(self, idx, val):
        self.range_set(idx, idx, val)
