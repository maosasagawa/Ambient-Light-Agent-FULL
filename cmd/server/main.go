package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"ambient-light-agent/internal/api"
	"ambient-light-agent/internal/config"
	"ambient-light-agent/internal/service"
	"ambient-light-agent/internal/storage"
	"ambient-light-agent/internal/ws"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	store, err := storage.NewStore(cfg.DataPath)
	if err != nil {
		log.Fatalf("init store: %v", err)
	}

	hub := ws.NewHub()
	go hub.Run()
	hardwareHub := ws.NewHardwareHub()

	knowledgeSvc, err := service.NewKnowledgeService(cfg.KnowledgeBasePath)
	if err != nil {
		log.Fatalf("init knowledge service: %v", err)
	}

	intentSvc := service.NewIntentService(cfg, store, knowledgeSvc)
	matrixSvc := service.NewMatrixService(cfg, store, hub)
	stripSvc := service.NewStripService(cfg, store, hub, knowledgeSvc)
	animSvc := service.NewAnimationService(cfg, store, hub)

	router := api.NewRouter(cfg, store, hub, hardwareHub, intentSvc, matrixSvc, stripSvc, animSvc)
	srv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           router,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		log.Printf("server listening on %s", cfg.ListenAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("shutdown error: %v", err)
	}
}
