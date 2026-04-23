# Vivado ILA capture loop for MAARS.
#
# Usage inside Vivado Tcl console (or vivado -mode tcl -source ...):
#   source vivado_ila_capture_to_csv.tcl
#   maars_ila_capture_loop <csv_path> <request_flag_path>
#
# Defaults when args are omitted:
#   csv_path         = ./ila_probe0.csv
#   request_flag_path= ./ila_capture_request.txt
#
# The script waits for request_flag_path to appear, captures from the first ILA,
# writes probe0 values to csv_path (one unsigned 32-bit integer per row), then
# deletes request_flag_path. Depth is forced to 16384.

namespace eval ::maars_ila {
    variable capture_depth 16384
    variable poll_ms 100
}

proc ::maars_ila::wait_for_request {request_flag_path poll_ms} {
    while {1} {
        if {[file exists $request_flag_path]} {
            return
        }
        after $poll_ms
    }
}

proc ::maars_ila::normalize_probe0_value {raw_value} {
    set value [string trim $raw_value]
    if {$value eq ""} {
        return ""
    }

    if {[regexp -nocase {^32'h([0-9a-f]+)$} $value -> hex_value]} {
        scan $hex_value %x decoded
        return $decoded
    }

    if {[regexp -nocase {^0x([0-9a-f]+)$} $value -> hex_value]} {
        scan $hex_value %x decoded
        return $decoded
    }

    if {[string is integer -strict $value]} {
        return $value
    }

    return ""
}

proc ::maars_ila::extract_probe0_csv {raw_csv_path out_csv_path} {
    set in [open $raw_csv_path r]
    set out [open $out_csv_path w]

    set header [gets $in]
    if {$header < 0} {
        close $in
        close $out
        return -code error "Raw ILA CSV is empty: $raw_csv_path"
    }

    set fields [split $header ","]
    set probe_idx -1
    for {set i 0} {$i < [llength $fields]} {incr i} {
        set f [string trim [lindex $fields $i] "\" "]
        if {[string match -nocase "*probe0*" $f]} {
            set probe_idx $i
            break
        }
    }

    if {$probe_idx < 0} {
        close $in
        close $out
        return -code error "Could not find probe0 column in $raw_csv_path"
    }

    while {[gets $in line] >= 0} {
        if {[string trim $line] eq ""} {
            continue
        }
        set cols [split $line ","]
        if {$probe_idx >= [llength $cols]} {
            continue
        }

        set normalized [::maars_ila::normalize_probe0_value [lindex $cols $probe_idx]]
        if {$normalized eq ""} {
            continue
        }
        puts $out $normalized
    }

    close $in
    close $out
}

proc ::maars_ila::capture_probe0_once {ila_obj csv_path} {
    variable capture_depth

    if {[catch {set_property CONTROL.DATA_DEPTH $capture_depth $ila_obj} err]} {
        puts "WARNING: failed to set ILA depth to $capture_depth: $err"
    }

    run_hw_ila $ila_obj
    wait_on_hw_ila $ila_obj

    set data_obj [upload_hw_ila_data $ila_obj]
    set raw_csv_path "${csv_path}.raw"

    if {[file exists $raw_csv_path]} {
        file delete -force $raw_csv_path
    }

    write_hw_ila_data -csv_file $raw_csv_path $data_obj
    ::maars_ila::extract_probe0_csv $raw_csv_path $csv_path

    if {[file exists $raw_csv_path]} {
        file delete -force $raw_csv_path
    }
}

proc maars_ila_capture_loop {{csv_path "./ila_probe0.csv"} {request_flag_path "./ila_capture_request.txt"}} {
    variable ::maars_ila::poll_ms

    open_hw_manager
    connect_hw_server
    open_hw_target

    set ilas [get_hw_ilas]
    if {[llength $ilas] == 0} {
        return -code error "No HW ILA cores found in the active target"
    }
    set ila_obj [lindex $ilas 0]

    puts "MAARS ILA capture loop started"
    puts "  ILA core: $ila_obj"
    puts "  CSV output: $csv_path"
    puts "  request flag: $request_flag_path"
    puts "  depth: $::maars_ila::capture_depth"

    while {1} {
        ::maars_ila::wait_for_request $request_flag_path $poll_ms

        if {[catch {::maars_ila::capture_probe0_once $ila_obj $csv_path} err]} {
            puts "ERROR: ILA capture failed: $err"
        } else {
            puts "Capture complete: wrote probe0 CSV to $csv_path"
        }

        if {[file exists $request_flag_path]} {
            file delete -force $request_flag_path
        }
    }
}
