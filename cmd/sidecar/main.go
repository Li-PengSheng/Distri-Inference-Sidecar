package main

import (
    "fmt"
    "net/http"
    "strings"
)

func main() {
    fmt.Println("Sidecar is running on :8080...")
    http.HandleFunc("/infer", func(w http.ResponseWriter, r *http.Request) {
        // 这里后续会加入 Dynamic Batching 逻辑
        // 目前先直接转发给 Python
        fmt.Fprint(w, "Sidecar received request and forwarding...")
    })
    http.ListenAndServe(":8080", nil)
}
