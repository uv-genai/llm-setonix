from pathlib import Path
from asyncinotify import Inotify, Mask
import asyncio

async def main():
    # Context manager to close the inotify handle after use
    with Inotify() as inotify:
        # Adding the watch can also be done outside of the context manager.
        # __enter__ doesn't actually do anything except return self.
        # This returns an asyncinotify.inotify.Watch instance
        inotify.add_watch("./SLURM_Logs", Mask.CLOSE | Mask.CLOSE_NOWRITE |
                                          Mask.ACCESS | Mask.MODIFY | 
                                          Mask.OPEN | Mask.CREATE | Mask.DELETE | 
                                          Mask.ATTRIB | Mask.CLOSE | Mask.MOVE | Mask.ONLYDIR)
        # Iterate events forever, yielding them one at a time
        async for event in inotify:
            # Events have a helpful __repr__.  They also have a reference to
            # their Watch instance.
            print(event)

            # the contained path may or may not be valid UTF-8.  See the note
            # below
            print(repr(event.path))

if __name__ == "__main__":
    asyncio.run(main())
