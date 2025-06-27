#!/bin/bash

export MOUNT_POINT=/home/paugamr/data
EXPECTED_SOURCE="EUBURN_FIRES@repo.sedoo.fr:/"
if mountpoint $MOUNT_POINT ; then
    echo "aeris disc mounted"
else
    (echo '8aVFFL#9Ez'; echo '') | sshfs EUBURN_FIRES@repo.sedoo.fr:/ $MOUNT_POINT -o password_stdin
fi

# Get the line from df that matches the mount point
MOUNT_INFO=$(df -h | grep -E "[[:space:]]$MOUNT_POINT$")

# Check if the mount point is found
if [ -z "$MOUNT_INFO" ]; then
    echo "❌ Mount point $MOUNT_POINT is not found."
    echo "umount and remount"
    umount $MOUNT_POINT
    (echo '8aVFFL#9Ez'; echo '') | sshfs EUBURN_FIRES@repo.sedoo.fr:/ $MOUNT_POINT -o password_stdin
fi

# Extract source (first column)
SOURCE=$(echo "$MOUNT_INFO" | awk '{print $1}')

# Compare with expected source
if [ "$SOURCE" != "$EXPECTED_SOURCE" ]; then
    echo "❌ Mount point $MOUNT_POINT is mounted, but source is '$SOURCE' (expected: '$EXPECTED_SOURCE')."
    exit 1
fi

