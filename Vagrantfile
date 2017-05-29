# -*- mode: ruby -*-
# vi: set ft=ruby :

# All Vagrant configuration is done below. The "2" in Vagrant.configure
# configures the configuration version (we support older styles for
# backwards compatibility). Please don't change it unless you know what
# you're doing.
Vagrant.configure("2") do |config|
  # The most common configuration options are documented and commented below.
  # For a complete reference, please see the online documentation at
  # https://docs.vagrantup.com.

  # Every Vagrant development environment requires a box. You can search for
  # boxes at https://atlas.hashicorp.com/search.
  config.vm.box = "ogarcia/archlinux-x64"

  # Disable automatic box update checking. If you disable this, then
  # boxes will only be checked for updates when the user runs
  # `vagrant box outdated`. This is not recommended.
  # config.vm.box_check_update = false

  # Create a forwarded port mapping which allows access to a specific port
  # within the machine from a port on the host machine. In the example below,
  # accessing "localhost:8080" will access port 80 on the guest machine.
  config.vm.network "forwarded_port", guest: 5501, host: 5501  # for executor service
  config.vm.network "forwarded_port", guest: 2200, host: 2200  # for gdb server

  # Create a private network, which allows host-only access to the machine
  # using a specific IP.
  #config.vm.network "private_network", type: "dhcp"

  # Create a public network, which generally matched to bridged network.
  # Bridged networks make the machine appear as another physical device on
  # your network.
  #config.vm.network "public_network", bridge: "eth0", type: "dhcp"

  # Share an additional folder to the guest VM. The first argument is
  # the path on the host to the actual folder. The second argument is
  # the path on the guest to mount the folder. And the optional third
  # argument is a set of non-required options.
  config.vm.synced_folder "/home/aetf", "/home/aetf"

  # Provider-specific configuration so you can fine-tune various
  # backing providers for Vagrant. These expose provider-specific options.
  # Example for VirtualBox:
  #
  # config.vm.provider "virtualbox" do |vb|
  #   # Display the VirtualBox GUI when booting the machine
  #   vb.gui = true
  #
  #   # Customize the amount of memory on the VM:
  #   vb.memory = "1024"
  # end
  #
  # View the documentation for the provider you are using for more
  # information on available options.

  # Define a Vagrant Push strategy for pushing to Atlas. Other push strategies
  # such as FTP and Heroku are also available. See the documentation at
  # https://docs.vagrantup.com/v2/push/atlas.html for more information.
  # config.push.define "atlas" do |push|
  #   push.app = "YOUR_ATLAS_USERNAME/YOUR_APPLICATION_NAME"
  # end

  # Enable provisioning with a shell script. Additional provisioners such as
  # Puppet, Chef, Ansible, Salt, and Docker are also available. Please see the
  # documentation for more information about their specific syntax and use.

  config.vm.provision "shell", inline: <<-SHELL
    sudo hostnamectl set-hostname arch-server
  SHELL

  config.vm.provision "file", source: "/etc/pacman.d/mirrorlist", destination: "mirrorlist"
  config.vm.provision "shell", inline: <<-SHELL
    sudo mv mirrorlist /etc/pacman.d/mirrorlist || echo "Update mirrorlist failed"
    sudo pacman -Syyu --noconfirm
    sudo pacman -S --noconfirm base-devel vim git cmake unzip
  SHELL

  config.vm.provision "shell", inline: <<-SHELL
    curl -JOL https://raw.githubusercontent.com/Aetf/Dotfiles/CC/home/bashrc
    mv bashrc $HOME/.bashrc
    echo "cd /vagrant" >> /home/vagrant/.bashrc
  SHELL

  config.vm.provision "file", source: "~/.gitconfig", destination: ".gitconfig"

  config.vm.provision "shell", inline: <<-SHELL
    mkdir tools buildbed && cd buildbed
    curl -JOL https://aur.archlinux.org/cgit/aur.git/snapshot/cower.tar.gz
    tar xvf cower.tar.gz
    pushd cower
    makepkg -s
    pacman -U --noconfirm cower*.pkg.tar.xz
    popd
    cower -d pacaur
    pushd pacaur
    makepkg -s
    pacman -U --noconfirm pacaur*.pkg.tar.xz
    popd
    pacaur -S --noconfirm downgrader
    downgrader protobuf
  SHELL
end
