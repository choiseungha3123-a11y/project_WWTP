package kr.kro.prjectwwtp.persistence;

import java.time.LocalDateTime;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.FakeDate;

public interface FakeDateRepository extends JpaRepository<FakeDate, LocalDateTime>{
	FakeDate findFirstByOrderByTodayDesc();
}