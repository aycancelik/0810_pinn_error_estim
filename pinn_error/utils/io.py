# Source - https://stackoverflow.com/a/16571630
# Posted by kindall, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-03, License - CC BY-SA 4.0

import sys
from io import StringIO


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout
