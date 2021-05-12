import os
import hypothesis

# Set a higher deadline on CI runs to allow for more extensive tests
hypothesis.settings.register_profile('ci', deadline=2000)

collect_ignore = ["benchmarking"]
