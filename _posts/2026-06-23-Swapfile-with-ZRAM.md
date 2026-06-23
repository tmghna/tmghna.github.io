---
layout: post
title: The Ultimate Guide to Btrfs Swapfile Hibernation Alongside ZRAM on Arch
description: For self record and reproducibility
authors: [Tamaghna]
categories: [Technical]
tags: [CachyOS, Linux, Arch, Snapshots, Error -5, ZRAM, swapfile, limine, AMD SME]
---

## Contents
1. [Explaining my motive](#explaining-my-motive)
2. [Introduction to the setup I worked on](#introduction-to-the-setup-i-worked-on)
3. [Snapshot trap](#snapshot-trap)
4. [The multiple fragmentation issue (Error -5)](#the-multiple-fragmentation-issue-error--5)
5. [Clarifying the CachyOS limine architecture](#clarifying-the-cachyos-limine-architecture)
6. [The silicon layer problem](#the-silicon-layer-problem)
7. [Alternative setup protocols](#alternative-setup-protocols) 

## Explaining my motive

The ultimate goal of this setup was simple: to preserve a highly customized, volatile workspace across hard power-offs. ZRAM is fantastic for keeping daily system performance blazing fast, but because it lives entirely in volatile memory, it vanishes the second the power is cut.

Hibernation mathematically requires a physical disk to write the RAM contents to. Therefore, a dedicated physical swapfile had to be created alongside ZRAM. The trick is to keep the disk swapfile at a very low priority (`0`) so the system ignores it during daily use, while keeping ZRAM at a high priority (`100`) to handle active swapping.

## Introduction to the setup I worked on

This specific configuration and troubleshooting journey took place on a modern, performance-focused Linux setup:

- **OS:** CachyOS (an Arch Linux derivative)
    
- **Filesystem:** Btrfs on a single NVMe drive
    
- **Window Manager:** Xmonad
    
- **Hardware:** AMD CPU/GPU Architecture
    
- **Bootloader:** Limine (CachyOS default)
    

## Snapshot trap

When setting up a swapfile on Btrfs, the immediate danger is **Timeshift**. Backup utilities like Timeshift take snapshots of the default root (`@`) and home (`@home`) subvolumes. If your swapfile is caught inside one of these snapshots, Btrfs forces the file into a "Copy-on-Write" (CoW) state. The Linux kernel will permanently refuse to resume from a CoW file, instantly breaking your hibernation pipeline.

To solve this, we must bypass the default subvolume and create a dedicated, isolated subvolume for the swapfile at the absolute top-level of the drive.

**The Fix:**


```bash
# 1. Mount the absolute raw root of the Btrfs drive
sudo mount -o subvolid=5 -U <YOUR_DRIVE_UUID> /mnt

# 2. Create the isolated subvolume (Timeshift ignores this)
sudo btrfs subvolume create /mnt/@swap
sudo umount /mnt

# 3. Create the mount point and update /etc/fstab
sudo mkdir -p /swap
```

Add this line to your `/etc/fstab` to mount it permanently:


```
UUID=<YOUR_DRIVE_UUID> /swap btrfs rw,noatime,subvol=/@swap 0 0
```

_(Run `sudo systemctl daemon-reload` and `sudo mount /swap` to activate it)._

## The multiple fragmentation issue (Error -5)

Once the subvolume is mounted, you generate the swapfile. However, if your SSD is somewhat full or internally fragmented, Btrfs might silently split your 16GB swapfile across multiple physical chunks (extents) on the disk.

The Linux kernel does not understand Btrfs during the wake-up phase; it only reads raw physical SSD blocks. If the file is split, the kernel reads the first chunk, hits garbage data, fails its internal checksum, and aborts the resume with a dreaded **EIO Error -5**.

**The Fix:** First, create the swapfile using the official Btrfs command:


```bash
sudo btrfs filesystem mkswapfile --size 16G /swap/swapfile
```

Check it for fragmentation:


```bash
sudo filefrag -v /swap/swapfile | grep extents
```

If it returns anything higher than `1 extent found`, you must delete the file and try again with a slightly smaller size (e.g., `14G`, `12G`, or `8G`). Because the kernel compresses RAM before hibernating, an 8GB contiguous swapfile is usually more than enough for 16GB of RAM.

Once you achieve exactly `1 extent`, activate it and add it to your `/etc/fstab`:


```bash
/swap/swapfile none swap defaults,pri=0 0 0
```

## Clarifying the CachyOS limine architecture

To tell the kernel where to look for the hibernation image, you need the physical offset of your un-fragmented swapfile:


```bash
sudo btrfs inspect-internal map-swapfile -r /swap/swapfile
```

Unlike standard Arch Linux, CachyOS dynamically generates its boot menu. If you manually edit `/boot/limine.conf`, your changes will be wiped out during the next kernel update. You must edit the Limine generator blueprint instead.

**The Fix:** Open `/etc/default/limine` and locate the `KERNEL_CMDLINE` string. You need to append your `resume` and `resume_offset` parameters here.

**Crucial Warning:** You must also change your mount flag from `rw` (Read-Write) to `ro` (Read-Only). If Btrfs mounts in `rw` during the wake-up phase, background tree-log tasks will trample and corrupt the hibernation image before it can be loaded into RAM.

Your kernel command line should look like this:


```ini
KERNEL_CMDLINE[default]+="... ro rootflags=subvol=/@ resume=UUID=<YOUR_UUID> resume_offset=<YOUR_OFFSET>"
```

Bake it in permanently:


```bash
sudo limine-mkinitcpio
```

## The silicon layer problem

Even with a perfect 1-extent swapfile and correct offsets, the kernel might successfully read the data but still fail to decompress it, throwing another Error -5. If you are on an AMD system, you are likely hitting the silicon layer trap: **Secure Memory Encryption (SME)** and **Zswap**.

Modern AMD processors generate a randomized hardware key to encrypt your RAM. When the laptop hibernates and powers off, that key is destroyed. Upon waking, the processor generates a _new_ key, making the saved RAM completely unintelligible gibberish.

**The Fix:** You must bypass these hardware and compression middlemen by adding kill-switches to your Limine blueprint.

Open `/etc/default/limine` and append these to your `KERNEL_CMDLINE`:


```ini
mem_encrypt=off zswap.enabled=0
```

Rebuild the boot menu (`sudo limine-mkinitcpio`), and your system will finally write plain, readable data to the disk.

## Alternative setup protocols

If your setup deviates from the CachyOS/AMD configuration outlined above, here is how you adapt the pipeline:

### Bootloader Variations

**GRUB (Standard Arch/Other Distros):**

- Add your parameters (`resume=UUID=... resume_offset=... ro`) to the `GRUB_CMDLINE_LINUX_DEFAULT` string inside `/etc/default/grub`.
    
- Run `sudo grub-mkconfig -o /boot/grub/grub.cfg` to bake it in.
    

**Standard Arch `mkinitcpio`:**

- If you are not on CachyOS, you must manually ensure the kernel is compiled with hibernation support.
    
- Open `/etc/mkinitcpio.conf` and add the word `resume` to your `HOOKS=(...)` array (it must be placed _after_ `udev` but _before_ `filesystems`).
    
- Run `sudo mkinitcpio -P` to generate your images.
    

### Hardware Variations

**Intel Processors:**

- Intel CPUs handle memory encryption differently (TME/MKTME), and it is rarely enabled by default in a way that breaks Linux hibernation. Intel users generally do not need to add the `mem_encrypt=off` flag.
    
- However, Intel users should still append `zswap.enabled=0` if they encounter data corruption during the wake-up phase.