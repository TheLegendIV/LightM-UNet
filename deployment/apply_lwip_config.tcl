# Run with: xsct apply_lwip_config.tcl
# Bulk-sets lwip211 BSP CONFIG.* properties so lwipopts.h regenerates with these
# values every time (instead of hand-editing the generated lwipopts.h, which gets
# wiped on every "Re-generate BSP Sources").

set ws_dir      [file join [file normalize [file dirname [info script]]] "FINN_interface"]
set xsa_file    [file join $ws_dir "zcu106_test" "hw" "top_wrapper.xsa"]
set platform_mss [file join $ws_dir "zcu106_test" "psu_cortexa53_0" "standalone_domain" "bsp" "system.mss"]

hsi::open_hw_design $xsa_file
hsi::open_sw_design $platform_mss
set lwip_lib [hsi::get_libs -filter {NAME == lwip211}]

set cfg {
    mem_size                 131072
    memp_n_pbuf               16
    memp_n_udp_pcb              4
    memp_n_tcp_pcb             32
    memp_n_tcp_pcb_listen       8
    memp_n_tcp_seg            512
    memp_n_sys_timeout          8
    memp_num_netbuf              8
    memp_num_netconn           16
    memp_num_api_msg           16
    memp_num_tcpip_msg          64
    pbuf_pool_size            256
    pbuf_pool_bufsize         1700
    pbuf_link_hlen              16
    arp_table_size              10
    arp_queueing               true
    icmp_ttl                   255
    ip_reassembly              true
    ip_frag                    true
    ip_reass_max_pbufs         128
    ip_frag_max_mtu            1500
    ip_default_ttl              255
    lwip_udp                   true
    udp_ttl                    255
    lwip_tcp                   true
    tcp_mss                   1460
    tcp_snd_buf               65535
    tcp_wnd                   65535
    tcp_ttl                    255
    tcp_maxrtx                  12
    tcp_synmaxrtx                4
    tcp_queue_ooseq            true
    lwip_tcp_keepalive         false
    no_sys_no_timers           false
    api_mode                RAW_API
}

foreach {name value} $cfg {
    common::set_property CONFIG.$name $value $lwip_lib
}

hsi::generate_bsp -dir [file join $ws_dir "zcu106_test" "psu_cortexa53_0" "standalone_domain" "bsp"] -proc psu_cortexa53_0

# TCP_SND_QUEUELEN has no CONFIG.* property -- lwip211.tcl always hardcodes it to
# "16 * TCP_SND_BUF/TCP_MSS", so patch it back in every time after regeneration.
set contrib_lwipopts [file join $ws_dir "zcu106_test" "psu_cortexa53_0" "standalone_domain" "bsp" \
    "psu_cortexa53_0" "libsrc" "lwip211_v1_8" "src" "contrib" "ports" "xilinx" "include" "lwipopts.h"]

set fd [open $contrib_lwipopts r]
set content [read $fd]
close $fd

set content [regsub {#define TCP_SND_QUEUELEN.*} $content \
    {#define TCP_SND_QUEUELEN (((TCP_SND_BUF + TCP_MSS - 1) / TCP_MSS) * 2)}]

set fd [open $contrib_lwipopts w]
puts -nonewline $fd $content
close $fd

puts "lwip211 CONFIG properties applied, BSP sources regenerated, TCP_SND_QUEUELEN patched."
